// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::DerefMut;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Barrier};

use arrayvec::ArrayVec;
use kvm_bindings::{
    KVM_MEM_READONLY, kvm_create_device, kvm_device_type_KVM_DEV_TYPE_VFIO,
    kvm_userspace_memory_region,
};
use vfio_bindings::bindings::vfio::*;
pub use vfio_ioctls::{
    VfioContainer, VfioDevice as InternalVfioDevice, VfioDeviceFd, VfioRegionInfoCap,
    VfioRegionInfoCapSparseMmap, VfioRegionSparseMmapArea,
};
use vm_allocator::{AddressAllocator, AllocPolicy, RangeInclusive};
use vm_memory::{GuestMemoryBackend, GuestMemoryRegion};
use vmm_sys_util::eventfd::EventFd;
use zerocopy::IntoBytes;

use crate::arch::host_page_size;
use crate::logger::{debug, error, warn};
use crate::pci::configuration::{
    BAR0_REG_IDX, Bars, COMMAND_BUS_MASTER_ENABLE, COMMAND_INTX_DISABLE, COMMAND_IO_SPACE_ENABLE,
    COMMAND_MEMORY_SPACE_ENABLE, COMMAND_REG, NUM_BAR_REGS, ROM_BAR_REG,
};
use crate::pci::msix::{MsixCap, MsixConfig};
use crate::pci::{PciCapabilityId, PciDevice, PciExpressCapabilityId, PciSBDF};
use crate::utils::{
    align_down_host_page, align_up_host_page, is_host_page_aligned, offset_from_lower_host_page,
    u64_to_usize, usize_to_u64,
};
use crate::vmm_config::device_passthrough::DevicePassthroughConfig;
use crate::vstate::bus::{BusDevice, BusError};
use crate::vstate::interrupts::InterruptError;
use crate::vstate::memory::{GuestMemoryMmap, GuestRegionType};
use crate::vstate::resources::ResourceAllocator;
use crate::vstate::vm::{KvmVm, VmError};

// Number of 4 byte registers in the config space
const PCI_CONFIG_SPACE_REGS: u16 = 1024;
// First BAR offset in the PCI config space.
const PCI_CONFIG_BAR_OFFSET: u32 = 0x10;
// Capability register offset in the PCI config space.
const PCI_CONFIG_CAPABILITY_OFFSET: u32 = 0x34;
// Extended capabilities register offset in the PCI config space.
const PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET: u16 = 0x100;
// IO BAR when first BAR bit is 1.
const PCI_CONFIG_IO_BAR: u32 = 1 << 0;
// 64-bit memory bar flag.
const PCI_CONFIG_MEMORY_BAR_64BIT: u32 = 1 << 2;
// Prefetchable BAR bit
const PCI_CONFIG_BAR_PREFETCHABLE: u32 = 1 << 3;

/// VfioError
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum VfioError {
    /// Failed to allocate guest address for BAR
    BarAllocation,
    /// Mmap failed: {0:?}
    Mmap(std::io::Error),
    /// Failed to allocate KVM slot
    KvmSlot,
    /// Failed to set KVM user memory region: {0}
    SetUserMemoryRegion(VmError),
    /// Failed to move MMIO bus range: {0}
    MoveBusRange(BusError),
    /// vfio-ioctls crate error: {0}
    VfioIoctls(#[from] vfio_ioctls::VfioError),
    /// Cannot create Msix vector group: {0}
    MsixConfig(#[from] InterruptError),
    /// Device does not provide MSIx irq
    NoMsixIrq,
    /// KVM failed to create KVM_DEV_TYPE_VFIO device: {0}
    KvmCreateVfioDevice(kvm_ioctls::Error),
    /// BAR{0} MSI-X table at offset {1:#x} size {2:#x} does not fit in region of size {3:#x}
    MsixTableOutOfRange(u8, u64, u64, u64),
    /// BAR{0} sparse mmap area at offset {1:#x} size {2:#x} does not fit in region of size {3:#x}
    SparseMmapAreaOutOfRange(u8, u64, u64, u64),
    /// BAR{0} sparse mmap area at gpa {1:#x} size {2:#x} overlaps MSI-X at gpa {3:#x} size {4:#x}
    SparseMmapAreaOverlapsMsixTable(u8, u64, u64, u64, u64),
}

/// Each used BAR has exactly one emulated area covering the whole BAR.
pub type VfioBarEmulatedAreas = ArrayVec<VfioBarEmulatedArea, { NUM_BAR_REGS as usize }>;

/// Description of the BAR area where reads/writes are emulated. Since mmapped areas take
/// precedence, only accesses to the parts of the BAR which are not mmapped (including the MSIx
/// table) are trapped.
#[derive(Debug, Copy, Clone)]
pub struct VfioBarEmulatedArea {
    bar_idx: u8,
    /// GPA of the area
    pub gpa: u64,
    /// Size of the area
    pub size: u64,
}

/// Wrapper around `Bars` types to automate dropping
#[derive(Debug)]
struct VfioBars {
    /// BARs as the guest sees them, with the addresses the guest wrote to their registers.
    guest_bars: Bars,
    /// BARs as the VMM uses them: the ranges handed out by the resource allocator, so they are the
    /// ones to give back on drop. They follow `guest_bars` on BAR relocation.
    vmm_bars: Bars,
    vm: Arc<KvmVm>,
}

impl VfioBars {
    fn new(
        vm: Arc<KvmVm>,
        bar_infos: &[VfioBarInfo; NUM_BAR_REGS as usize],
    ) -> Result<Self, VfioError> {
        let bars = {
            let mut resource_allocator_lock = vm.resource_allocator();
            let resource_allocator = resource_allocator_lock.deref_mut();
            vfio_allocate_memory_ranges_for_bars(resource_allocator, bar_infos)?
        };
        Ok(Self {
            guest_bars: bars,
            vmm_bars: bars,
            vm,
        })
    }
}

impl Drop for VfioBars {
    fn drop(&mut self) {
        let mut resource_allocator_lock = self.vm.resource_allocator();
        let resource_allocator = resource_allocator_lock.deref_mut();
        vfio_deallocate_memory_ranges_for_bars(resource_allocator, &self.vmm_bars);
    }
}

/// Information about the bar mapping
#[derive(Debug, Copy, Clone)]
struct VfioBarMapping {
    kvm_slot: u32,
    kvm_flags: u32,
    gpa: u64,
    size: u64,
    hva: u64,
}

/// Container for everything MSIx related
#[derive(Debug)]
struct VfioMsixState {
    /// Register idx where the capability is in the configuration space
    register: u8,
    cap: MsixCap,
    config: MsixConfig,
}

/// Mask for specific register in the configuration space
#[derive(Debug)]
struct VfioRegisterMask {
    /// Register id to mask
    register: u16,
    /// Mask for bits which should remain intact.
    /// Applied as (R & mask) | value
    mask: u32,
    /// Value to use with the mask
    value: u32,
}

/// The VFIO device information
pub struct VfioDevice {
    /// Configuration with which the device was created
    pub config: DevicePassthroughConfig,
    /// SBDF of the device in the configuration space
    pub sbdf: PciSBDF,
    device: InternalVfioDevice,
    bars: VfioBars,
    bar_mappings: Vec<VfioBarMapping>,
    /// Areas of the BARs which are emulated instead of being mapped into the guest.
    pub emulated_areas: VfioBarEmulatedAreas,
    msix_state: VfioMsixState,
    masks: Vec<VfioRegisterMask>,
    vm: Arc<KvmVm>,
}

impl std::fmt::Debug for VfioDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VfioDevice")
            .field("config", &self.config)
            .field("sbdf", &self.sbdf)
            .finish()
    }
}

impl VfioDevice {
    /// Create new VfioDevice
    pub fn new(
        container: &Arc<VfioContainer>,
        vm: &Arc<KvmVm>,
        config: DevicePassthroughConfig,
        sbdf: PciSBDF,
    ) -> Result<VfioDevice, VfioError> {
        vfio_init_device(container, vm, config, sbdf)
    }

    fn maybe_relocate_all_bars(&mut self) {
        vfio_maybe_relocate_all_bars(self);
    }
}

impl Drop for VfioDevice {
    fn drop(&mut self) {
        vfio_deinit_device(self);
    }
}

// Internal type for distributing the emulated area accesses
// between different components
#[derive(Debug)]
enum HandleBarAccessResult {
    // Access partially overlaps MSIx table
    PartialOverlap,
    // Need to access MSIx table at some offset
    MsixTable(u64),
    // Need to pass access to the device to the specific BAR at specific offset
    Device { bar_idx: u8, offset: u64 },
}

// Distribute BAR access aiming at the BAR containing MSIx table.
fn vfio_distribute_msix_access(
    emulated_area: &VfioBarEmulatedArea,
    msix_cap: &MsixCap,
    base: u64,
    offset: u64,
    data_len: u64,
) -> HandleBarAccessResult {
    // SAFETY: if this is ever hit it would mean we have a bug in the code which adds
    // VfioBarEmulatedArea into the MmioBus.
    assert_eq!(emulated_area.gpa, base);

    let data_start = offset;
    let data_end = offset + data_len;

    // Layers above must ensure this
    assert!(data_end <= emulated_area.size);

    let (t_off, t_size) = msix_cap.table_bar_offset_and_size();
    let (t_start, t_size) = (t_off as u64, t_size as u64);
    let t_end = t_start + t_size;
    if t_start <= data_start && data_end <= t_end {
        return HandleBarAccessResult::MsixTable(offset - t_start);
    }
    // Reject partial overlap with table.
    // This should not happen in normal operations, but malicious
    // driver can try this.
    // In this case it should be fine to ignore the access altogether
    if data_start < t_end && t_start < data_end {
        return HandleBarAccessResult::PartialOverlap;
    }

    HandleBarAccessResult::Device {
        bar_idx: emulated_area.bar_idx,
        offset,
    }
}

// Distribute BAR access aiming at the emulated areas
fn vfio_distribute_emulated_access(
    emulated_areas: &[VfioBarEmulatedArea],
    msix_cap: &MsixCap,
    base: u64,
    offset: u64,
    data_len: u64,
) -> HandleBarAccessResult {
    for area in emulated_areas.iter() {
        if area.gpa == base {
            let result = if area.bar_idx == msix_cap.table_bir() {
                vfio_distribute_msix_access(area, msix_cap, base, offset, data_len)
            } else {
                HandleBarAccessResult::Device {
                    bar_idx: area.bar_idx,
                    offset,
                }
            };
            return result;
        }
    }
    // All accesses must land within an emulated area
    unreachable!();
}

impl BusDevice for VfioDevice {
    fn read(&mut self, base: u64, offset: u64, data: &mut [u8]) {
        match vfio_distribute_emulated_access(
            &self.emulated_areas,
            &self.msix_state.cap,
            base,
            offset,
            usize_to_u64(data.len()),
        ) {
            HandleBarAccessResult::PartialOverlap => {
                warn!(
                    "[{}] BusDevice::read ignoring read with partial overlap: base: {base:#x} \
                     offset: {offset:#x}",
                    self.config.id
                );
                data.fill(0);
            }
            HandleBarAccessResult::MsixTable(offset) => {
                self.msix_state.config.read_table(offset, data);
            }
            HandleBarAccessResult::Device { bar_idx, offset } => {
                let region_size = self.device.get_region_size(bar_idx as u32);
                // If this ever fires, we have a bug in emulated area calculation code since
                // the area expands past the region end
                assert!(offset + usize_to_u64(data.len()) <= region_size);
                self.device.region_read(bar_idx as u32, data, offset);
            }
        }
    }

    fn write(&mut self, base: u64, offset: u64, data: &[u8]) -> Option<Arc<Barrier>> {
        match vfio_distribute_emulated_access(
            &self.emulated_areas,
            &self.msix_state.cap,
            base,
            offset,
            usize_to_u64(data.len()),
        ) {
            HandleBarAccessResult::PartialOverlap => {
                warn!(
                    "[{}] BusDevice::write ignoring write with partial overlap: base: {base:#x} \
                     offset: {offset:#x}",
                    self.config.id
                );
            }
            HandleBarAccessResult::MsixTable(offset) => {
                self.msix_state.config.write_table(offset, data);
            }
            HandleBarAccessResult::Device { bar_idx, offset } => {
                let region_size = self.device.get_region_size(bar_idx as u32);
                // If this ever fires, we have a bug in emulated area calculation code since
                // the area expands past the region end
                assert!(offset + usize_to_u64(data.len()) <= region_size);
                self.device.region_write(bar_idx as u32, data, offset);
            }
        }
        None
    }
}

fn vfio_device_memory_space_enabled(device: &InternalVfioDevice) -> bool {
    let command_reg_offset = COMMAND_REG as u64 * 4;
    let mut result: u32 = 0;
    device.region_read(
        VFIO_PCI_CONFIG_REGION_INDEX,
        result.as_mut_bytes(),
        command_reg_offset,
    );
    result & COMMAND_MEMORY_SPACE_ENABLE != 0
}

fn vfio_device_reset_command_reg(device: &InternalVfioDevice) {
    let command_reg_offset = COMMAND_REG as u64 * 4;
    let mut reg: u32 = 0;
    device.region_read(
        VFIO_PCI_CONFIG_REGION_INDEX,
        reg.as_mut_bytes(),
        command_reg_offset,
    );
    // Reset I/O space, memory space, bus master and INTx disable bits to their default
    // value of 0. This is the state the guest expects the device to be in during init.
    reg &= !(COMMAND_IO_SPACE_ENABLE
        | COMMAND_MEMORY_SPACE_ENABLE
        | COMMAND_BUS_MASTER_ENABLE
        | COMMAND_INTX_DISABLE);

    // Only write the command register, the status register has write-1-to-clear bits
    device.region_write(
        VFIO_PCI_CONFIG_REGION_INDEX,
        &reg.as_bytes()[..2],
        command_reg_offset,
    );
}

impl PciDevice for VfioDevice {
    fn write_config_register(
        &mut self,
        reg_idx: u16,
        offset: u8,
        data: &[u8],
    ) -> Option<Arc<Barrier>> {
        let mut handled: bool = false;
        if BAR0_REG_IDX <= reg_idx && reg_idx < BAR0_REG_IDX + u16::from(NUM_BAR_REGS) {
            // reg_idx is in [BAR0_REG, BAR0_REG+NUM_BAR_REGS), so the difference is 0..5.
            #[allow(clippy::cast_possible_truncation)]
            let bar_idx = (reg_idx - BAR0_REG_IDX) as u8;
            // offset is within a 4-byte PCI config register (0..3).
            self.bars.guest_bars.write(bar_idx, offset, data);

            // With memory decoding disabled, relocation happens once it is enabled
            // Also skip writes for the BAR size query dance
            if vfio_device_memory_space_enabled(&self.device) && data != [0xff; 4] {
                let bar_idx = if self.bars.vmm_bars.bar_idx_valid(bar_idx) {
                    bar_idx
                } else {
                    bar_idx - 1
                };
                vfio_maybe_relocate_bar(
                    &mut self.bars,
                    bar_idx,
                    &mut self.emulated_areas,
                    &mut self.bar_mappings,
                    &self.vm,
                );
            }

            // write is passed through to the kernel in case it needs to know
            // about them.
        } else if reg_idx == ROM_BAR_REG {
            // We don't support ROM BAR
            handled = true;
            warn!(
                "[{}] PciDevice::write_config_register ignoring write to the ROM BAR: offset: \
                 {offset:#x}",
                self.config.id
            );
        } else if reg_idx == u16::from(self.msix_state.register) {
            // offset is within a 4-byte PCI config register (0..3).
            self.msix_state.config.write_msg_ctl_register(offset, data);
            // Don't set `handled` since we need to passthrough write
            // to the msg_ctl register to the device, so it will enable Msix
            // interrupts
        } else if reg_idx == COMMAND_REG {
            // This is special write that can force us to relocate BARs
            handled = true;

            let mem_enabled_before = vfio_device_memory_space_enabled(&self.device);

            let config_offset = reg_idx as u64 * 4 + offset as u64;
            self.device
                .region_write(VFIO_PCI_CONFIG_REGION_INDEX, data, config_offset);

            if !mem_enabled_before && vfio_device_memory_space_enabled(&self.device) {
                self.maybe_relocate_all_bars();
            }
        } else {
            // If we mask some registers, there is no reason to allow writing to them
            handled |= self.masks.iter().any(|m| m.register == reg_idx);
        }
        if !handled {
            let config_offset = reg_idx as u64 * 4 + offset as u64;
            self.device
                .region_write(VFIO_PCI_CONFIG_REGION_INDEX, data, config_offset);
        }
        None
    }
    fn read_config_register(&mut self, reg_idx: u16) -> u32 {
        let mut result: u32 = 0;
        if BAR0_REG_IDX <= reg_idx && reg_idx < BAR0_REG_IDX + u16::from(NUM_BAR_REGS) {
            // reg_idx is in [BAR0_REG, BAR0_REG+NUM_BAR_REGS), so the difference is 0..5.
            #[allow(clippy::cast_possible_truncation)]
            let bar_idx = (reg_idx - BAR0_REG_IDX) as u8;
            self.bars.guest_bars.read(bar_idx, 0, result.as_mut_bytes());
        } else if reg_idx == ROM_BAR_REG {
            // We don't support ROM BAR
            warn!(
                "[{}] PciDevice::read_config_register ignoring read to the ROM BAR",
                self.config.id
            );
        } else {
            let config_offset = reg_idx as u64 * 4;
            self.device.region_read(
                VFIO_PCI_CONFIG_REGION_INDEX,
                result.as_mut_bytes(),
                config_offset,
            );
            if reg_idx == u16::from(self.msix_state.register) {
                // Since we emulate the MsixCap, we need to set the Mask and Msix enable bits to
                // values we have, and not what device has.
                let msg_ctl = self.msix_state.config.as_msg_ctl();
                result &= 0x0000ffff;
                result |= u32::from(msg_ctl) << 16;
            }
            if let Some(mask) = self.masks.iter().find(|mask| mask.register == reg_idx) {
                result = (result & mask.mask) | mask.value;
            }
        }
        result
    }
}

/// Go through the PCI config space and reads all legacy and PCIe capabilities. Find the MSIx
/// cap if present and construct an array of masks which should be used to overwrite values in the
/// config space during read operations.
fn vfio_get_pci_capabilities(
    config_space: &[u32; PCI_CONFIG_SPACE_REGS as usize],
) -> (Option<(MsixCap, u8)>, Vec<VfioRegisterMask>) {
    let (msix_cap_and_register, has_pci_express_cap) = vfio_read_legacy_caps(config_space);

    // PCIe extended capabilities only exist if the device exposes a PCI Express capability.
    let masks = if has_pci_express_cap {
        vfio_read_extended_caps(config_space)
    } else {
        Vec::new()
    };

    (msix_cap_and_register, masks)
}

/// Minor utility function to read `bytes.len()` bytes from `config_space` starting at `offset`.
fn vfio_config_space_read_bytes(
    config_space: &[u32; PCI_CONFIG_SPACE_REGS as usize],
    offset: u32,
    bytes: &mut [u8],
) {
    let reg_idx = offset / 4;
    let in_reg_offset = offset % 4;
    let reg = config_space[reg_idx as usize];
    bytes.copy_from_slice(&reg.as_bytes()[in_reg_offset as usize..][0..bytes.len()]);
}

/// Walk the legacy PCI capability list. Return the MSI-X capability together with the register it
/// resides in (if present) and whether the device exposes a PCI Express capability.
fn vfio_read_legacy_caps(
    config_space: &[u32; PCI_CONFIG_SPACE_REGS as usize],
) -> (Option<(MsixCap, u8)>, bool) {
    let mut next_cap_offset: u8 = 0;
    vfio_config_space_read_bytes(
        config_space,
        PCI_CONFIG_CAPABILITY_OFFSET,
        next_cap_offset.as_mut_bytes(),
    );
    debug!("PCI CAPS offset: {}", next_cap_offset);

    let mut msix_cap_and_register = None;
    let mut has_pci_express_cap = false;
    // The legacy region with PCI cap is 256 bytes long and
    // split into 4 byte registers.
    const LOOP_UPPER_BOUND: u32 = 256 / 4;
    let mut loop_counter: u32 = 0;
    while next_cap_offset != 0 && loop_counter < LOOP_UPPER_BOUND {
        loop_counter += 1;

        // clear low 2 bits just in case to get 4 byte aligned address
        next_cap_offset &= 0xfc;

        let mut cap_id_and_next_ptr: u16 = 0;
        vfio_config_space_read_bytes(
            config_space,
            next_cap_offset as u32,
            cap_id_and_next_ptr.as_mut_bytes(),
        );

        let current_cap_offset = next_cap_offset;

        // PCIe spec revision 6.0: 7.5.3.1 PCI Express Capability List Register
        // |       32-16     |          15-8           |      7-0      |
        // |   Cap register  | Next Capability Pointer | Capability ID |
        let cap_id: u8 = (cap_id_and_next_ptr & 0xff) as u8;
        next_cap_offset = ((cap_id_and_next_ptr & 0xff00) >> 8) as u8;
        debug!("PCI CAP id: {cap_id} next offset: {next_cap_offset:#x}");

        let cap = PciCapabilityId::from(cap_id);
        let register = current_cap_offset / 4;
        debug!("Found pci cap: {cap:?} at offset: {current_cap_offset:#x}({register})");

        match cap {
            PciCapabilityId::PciExpress => {
                has_pci_express_cap = true;
            }
            PciCapabilityId::MsiX => {
                // PCIe spec revision 6.0: 7.7.2 MSI-X Capability and Table Structure
                let mut msg_ctl: u16 = 0;
                let mut table: u32 = 0;
                let mut pba: u32 = 0;
                vfio_config_space_read_bytes(
                    config_space,
                    (current_cap_offset as u32) + 2,
                    msg_ctl.as_mut_bytes(),
                );
                vfio_config_space_read_bytes(
                    config_space,
                    (current_cap_offset as u32) + 4,
                    table.as_mut_bytes(),
                );
                vfio_config_space_read_bytes(
                    config_space,
                    (current_cap_offset as u32) + 8,
                    pba.as_mut_bytes(),
                );
                msix_cap_and_register = Some((
                    MsixCap {
                        msg_ctl,
                        table,
                        pba,
                    },
                    register,
                ));
            }
            _ => {
                // We don't mask legacy PCI capabilities, so all of them will be presented to the
                // guest.
            }
        };
    }

    (msix_cap_and_register, has_pci_express_cap)
}

/// Walk the PCIe extended capability list and construct an array of masks which should be used to
/// overwrite values in the config space during read operations.
fn vfio_read_extended_caps(
    config_space: &[u32; PCI_CONFIG_SPACE_REGS as usize],
) -> Vec<VfioRegisterMask> {
    let mut masks = Vec::new();
    let mut next_cap_offset: u16 = PCI_CONFIG_EXTENDED_CAPABILITY_OFFSET;

    // The PCIe region is 4K in size and split into 4 byte registers
    const LOOP_UPPER_BOUND: u16 = PCI_CONFIG_SPACE_REGS;
    let mut loop_counter: u16 = 0;
    while next_cap_offset != 0 && loop_counter < LOOP_UPPER_BOUND {
        loop_counter += 1;

        // clear high 4 bits and low 2 bits to get 12 bit value of 4 byte aligned address
        next_cap_offset &= 0x0ffc;

        let mut cap_id_and_next_ptr: u32 = 0;
        vfio_config_space_read_bytes(
            config_space,
            next_cap_offset as u32,
            cap_id_and_next_ptr.as_mut_bytes(),
        );

        let current_cap_offset = next_cap_offset;

        // PCIe spec revision 6.0: 7.7.3.1 Secondary PCI Express Extended Capability Header
        // |           31-20        |         19-16       |          15-0         |
        // | Next capability offset | Capability Version  |   PCIe Capability ID  |
        let cap_id: u16 = (cap_id_and_next_ptr & 0xffff) as u16;
        next_cap_offset = (cap_id_and_next_ptr >> 20) as u16;

        let pci_cap = PciExpressCapabilityId::from(cap_id);
        let register = current_cap_offset / 4;
        debug!("Found pci ext cap: {pci_cap:?} cap at offset: {current_cap_offset:#x}({register})");

        // Find registers which contain the headers of PCIe caps we want to filter out of the
        // capability list. The "filtering" is done by changing the "PCI Express Cap ID" part of
        // the register to 0 which represents the "null" capability in the PCIe spec. The actual
        // chain of capabilities is not broken by this action. When guest driver encounters this
        // capability it just jumps to the next one since the "Next Cap Pointer" is intact.
        //
        // NOTE: the list of capabilities is hard-coded for now. In the future this may be
        // configurable from the user side.
        match pci_cap {
            // Mask ARI since we don't implement it
            PciExpressCapabilityId::AlternativeRoutingIdentificationInterpretation |
            // Mask ReBAR since we don't implement it
            PciExpressCapabilityId::ResizeableBar |
            // Mask SR-IOV since it should not be exposed to the VM and it contains host
            // physical addresses of BARs.
            PciExpressCapabilityId::SingleRootIoVirtualization => {
                debug!(
                    "Found {pci_cap:?} cap to be masked: {register}({current_cap_offset:#x})"
                );
                masks.push(VfioRegisterMask {
                    register,
                    mask: 0xffff_0000,
                    value: 0x0000_0000,
                });
            }
            _ => {
                // Rest of PCI Extended capabilities are presented to the guest.
            }
        }
    }

    masks
}

/// Allocate memory ranges for BARs from mmio32 or mmio64 allocators.
fn vfio_allocate_memory_ranges_for_bars(
    resource_allocator: &mut ResourceAllocator,
    bar_infos: &[VfioBarInfo; NUM_BAR_REGS as usize],
) -> Result<Bars, VfioError> {
    let mut bars = Bars::default();
    let mut bar_idx = 0;
    let host_page_size = usize_to_u64(host_page_size());
    while bar_idx < NUM_BAR_REGS {
        let bar_value = bar_infos[bar_idx as usize].bar_value;
        let size = bar_infos[bar_idx as usize].size;

        let is_io_bar = bar_value & PCI_CONFIG_IO_BAR != 0;
        let is_64_bits = bar_value & PCI_CONFIG_MEMORY_BAR_64BIT != 0;
        let is_prefetchable = bar_value & PCI_CONFIG_BAR_PREFETCHABLE != 0;

        if is_64_bits && bar_idx == NUM_BAR_REGS - 1 {
            warn!("BAR{bar_idx} is last BAR but marked as 64bit. Skipping");
            break;
        }

        // This checks both size being power of 2 and size != 0
        if size.is_power_of_two() {
            // PCIe spec revision 6.0: 7.5.1.2.1 Base Address Registers
            // This design implies that all address spaces used are a power of two
            // in size and are naturally aligned.
            let alignment = std::cmp::max(host_page_size, size);

            let gpa;
            if is_io_bar {
                warn!(
                    "BAR{bar_idx} size: {size:>#10x} io_bar: {is_io_bar} 64bits: {is_64_bits} \
                     prefetchable: {is_prefetchable} Skipping IO BAR"
                );
                bar_idx += 1;
                continue;
            }

            if is_64_bits && is_prefetchable {
                match resource_allocator.mmio64_memory.allocate(
                    size,
                    alignment,
                    AllocPolicy::FirstMatch,
                ) {
                    Ok(range) => {
                        gpa = range.start();
                        bars.set_bar_64(bar_idx, gpa, size, is_prefetchable.into());
                    }
                    Err(_) => {
                        vfio_deallocate_memory_ranges_for_bars(resource_allocator, &bars);
                        return Err(VfioError::BarAllocation);
                    }
                }
            } else {
                match resource_allocator.mmio32_memory.allocate(
                    size,
                    alignment,
                    AllocPolicy::FirstMatch,
                ) {
                    Ok(range) => {
                        gpa = range.start();
                        // SAFETY: both `gpa` and `size` went through 32bit resource allocator,
                        // so they must fit in u32
                        #[allow(clippy::cast_possible_truncation)]
                        bars.set_bar_32(bar_idx, gpa as u32, size as u32, is_prefetchable.into());
                    }
                    Err(_) => {
                        vfio_deallocate_memory_ranges_for_bars(resource_allocator, &bars);
                        return Err(VfioError::BarAllocation);
                    }
                }
            }
            debug!(
                "BAR{bar_idx} gpa: [{:#x}..{:#x}] size: {size:>#10x} io_bar: {is_io_bar} 64bits: \
                 {is_64_bits} prefetchable: {is_prefetchable}",
                gpa,
                gpa + size
            );
        } else {
            if size == 0 {
                debug!("BAR{bar_idx} has 0 size. Skipping");
            } else {
                warn!("BAR{bar_idx} has non power of 2 size: {size}. Skipping");
            }
        }
        if is_64_bits {
            bar_idx += 1;
        }
        bar_idx += 1;
    }
    Ok(bars)
}

/// 64bit BARs can be placed in either MMIO window, so the allocator is picked by the address.
fn vfio_bar_allocator(ra: &mut ResourceAllocator, addr: u64) -> &mut AddressAllocator {
    if addr < ra.mmio64_memory.base() {
        &mut ra.mmio32_memory
    } else {
        &mut ra.mmio64_memory
    }
}

/// Give memory ranges allocated for BARs back to the resource allocator
fn vfio_deallocate_memory_ranges_for_bars(resource_allocator: &mut ResourceAllocator, bars: &Bars) {
    let mut bar_idx = 0;
    while bar_idx < NUM_BAR_REGS {
        if bars.bars[bar_idx as usize].used() {
            let start = bars.get_bar_addr(bar_idx);
            let size = bars.get_bar_size(bar_idx);

            // SAFETY: these values were provided by the allocator in the first place
            let range = RangeInclusive::new(start, start + size - 1).unwrap();
            let allocator = vfio_bar_allocator(resource_allocator, start);
            allocator.free(&range).unwrap();
            if bars.bars[bar_idx as usize].is_64bit() {
                bar_idx += 1;
            }
        }
        bar_idx += 1;
    }
}

/// Internal type to store vfio region info from the kernel.
/// Each region corresponds to a single BAR register in the config space (6 in total).
///
/// The `size` parameter represents the size of the `logical` BAR, meaning if BAR0 is 64-bit,
/// then `VfioBarInfo[0]` will have the full 64-bit size, and `VfioBarInfo[1]` will have size 0.
///
/// To determine if the BAR is 32- or 64-bit, use `bar_value` which is a raw value obtained from the
/// config space.
#[derive(Debug, Clone)]
struct VfioBarInfo {
    bar_value: u32,
    flags: u32,
    size: u64,
    offset: u64,
    caps: Vec<VfioRegionInfoCap>,
}

/// Internal type to store areas needed to be mmaped for the device
#[derive(Debug, Clone, Copy)]
struct VfioBarMappableArea {
    gpa: u64,
    /// Offset to use when mmapping the vfio device fd in order to get the needed part of the
    /// region/BAR
    vfio_fd_offset: u64,
    size: u64,
    prot: i32,
}

/// Check if ranges overlap each other (touching does not count)
fn vfio_ranges_overlap(start_a: u64, size_a: u64, start_b: u64, size_b: u64) -> bool {
    start_a.max(start_b) < (start_a + size_a).min(start_b + size_b)
}

/// Calculate different areas of BARs of a device:
/// - every BAR gets its own emulated area, so all guest accesses which are not handled by mmapped
///   areas go through our dispatch code and we do read/write operations on the guest's behalf
/// - mmappable areas of BARs will be `mmap`ed on top of the emulated areas and passed through
///   directly to the guest without any emulation on our side. Since KVM memory slots take
///   precedence, accesses to them are not trapped.
///
/// As an example, the BAR holding the MSIx table is split into this arrangement:
///
/// [ mmapped area ][ emulated MSIx table page ][ mmapped area ]
///
/// where each `area` is host page aligned and only the MSIx table page emulated.
///
/// BARs which the kernel does not allow us to mmap (or the page with the MSIx table in them) or
/// which are smaller than the host page size have no mmappable areas.
fn vfio_calculate_bar_areas(
    bars: &Bars,
    bar_infos: &[VfioBarInfo; NUM_BAR_REGS as usize],
    msix_cap: &MsixCap,
) -> Result<(Vec<VfioBarMappableArea>, VfioBarEmulatedAreas), VfioError> {
    // There are 6 BARs with maximum of 1 MSIx table gap, so the maximum number of mappable areas
    // is 7. The only reason to use `Vec` instead of `ArrayVec` here is that this vector can be
    // populated from the `sparse_mmap_cap` which can contain a different number of areas. But
    // in any case the size here is limited by the `nr_areas` field in the
    // `vfio_region_info_cap_sparse_mmap` struct. This field has the `u32` type.
    let mut mmappable_areas = Vec::with_capacity(7);
    let mut emulated_areas = VfioBarEmulatedAreas::new();
    let mut bar_idx: u8 = 0;

    let host_page_size = usize_to_u64(host_page_size());
    while bar_idx < NUM_BAR_REGS {
        if bars.bars[bar_idx as usize].used() {
            let bar_gpa = bars.get_bar_addr(bar_idx);
            let bar_size = bars.get_bar_size(bar_idx);
            let bar_info = &bar_infos[bar_idx as usize];

            let can_mmap = bar_info.flags & VFIO_REGION_INFO_FLAG_MMAP != 0;
            let mut has_msix_mappable = false;
            let mut sparse_mmap_cap = None;
            for cap in bar_info.caps.iter() {
                match cap {
                    VfioRegionInfoCap::SparseMmap(cap) => sparse_mmap_cap = Some(cap),
                    VfioRegionInfoCap::MsixMappable => has_msix_mappable = true,
                    _ => {}
                }
            }

            let mut msix_table_offset = 0;
            let mut msix_table_size = 0;
            let mut msix_table_gpa = 0;
            let contains_msix_table = bar_idx == msix_cap.table_bir();

            if contains_msix_table {
                let (offset, size) = msix_cap.table_bar_offset_and_size();
                // Since original `offset` and `size` are `u32` and `u16`, their addition
                // cannot overflow when widened to `u64`;
                let (offset, size) = (offset as u64, size as u64);

                if bar_size < offset + size {
                    return Err(VfioError::MsixTableOutOfRange(
                        bar_idx, offset, size, bar_size,
                    ));
                }

                let offset_in_area = offset_from_lower_host_page(offset);
                msix_table_offset = align_down_host_page(offset);
                msix_table_size = align_up_host_page(offset_in_area + size);
                msix_table_gpa = bar_gpa + msix_table_offset;
            }

            let mut prot = 0;
            if bar_info.flags & VFIO_REGION_INFO_FLAG_READ != 0 {
                prot |= libc::PROT_READ;
            }
            if bar_info.flags & VFIO_REGION_INFO_FLAG_WRITE != 0 {
                prot |= libc::PROT_WRITE;
            }

            debug!(
                "BAR{bar_idx} emulated area: [{bar_gpa:#x}..{:#x}]",
                bar_gpa + bar_size
            );
            emulated_areas.push(VfioBarEmulatedArea {
                bar_idx,
                gpa: bar_gpa,
                size: bar_size,
            });

            let bar_areas_start = mmappable_areas.len();
            if host_page_size <= bar_info.size && can_mmap {
                if let Some(cap) = sparse_mmap_cap {
                    for area in cap.areas.iter() {
                        // Even though these are kernel provided values, do additional
                        // sanity checks.
                        if area
                            .offset
                            .checked_add(area.size)
                            .is_none_or(|end| bar_size < end)
                            || area.size == 0
                            || !is_host_page_aligned(area.offset)
                            || !is_host_page_aligned(area.size)
                        {
                            return Err(VfioError::SparseMmapAreaOutOfRange(
                                bar_idx,
                                area.offset,
                                area.size,
                                bar_size,
                            ));
                        }
                        // The kernel is expected to exclude the MSI-X table from the sparse
                        // mmap areas. If it did not, `mmap`ing the area would pass the table
                        // through to the guest while we also emulate it, which would let the
                        // guest program interrupts directly.
                        let gpa = bar_gpa + area.offset;
                        if contains_msix_table
                            && vfio_ranges_overlap(gpa, area.size, msix_table_gpa, msix_table_size)
                        {
                            return Err(VfioError::SparseMmapAreaOverlapsMsixTable(
                                bar_idx,
                                gpa,
                                area.size,
                                msix_table_gpa,
                                msix_table_size,
                            ));
                        }
                        mmappable_areas.push(VfioBarMappableArea {
                            gpa,
                            vfio_fd_offset: bar_info.offset + area.offset,
                            size: area.size,
                            prot,
                        });
                    }
                } else if has_msix_mappable || !contains_msix_table {
                    if contains_msix_table {
                        // If the table is within the region, then the page aligned MSIx table
                        // area must be within the BAR.
                        assert!(msix_table_offset + msix_table_size <= bar_size);
                    }
                    // There can only be one gap in the BAR,
                    // so the maximum number of mmappable areas is 2.
                    //
                    // The most advanced case will look like this:
                    //
                    // region start           region end
                    //      [ area ][ gap ][ last area ]
                    //     low                      high
                    //
                    if msix_table_offset != 0 {
                        mmappable_areas.push(VfioBarMappableArea {
                            gpa: bar_gpa,
                            vfio_fd_offset: bar_info.offset,
                            size: msix_table_offset,
                            prot,
                        });
                    }
                    let offset = msix_table_offset + msix_table_size;
                    let last_area_size = bar_size - offset;
                    if last_area_size != 0 {
                        mmappable_areas.push(VfioBarMappableArea {
                            gpa: bar_gpa + offset,
                            vfio_fd_offset: bar_info.offset + offset,
                            size: last_area_size,
                            prot,
                        });
                    }
                }
            }
            for area in mmappable_areas[bar_areas_start..].iter() {
                debug!(
                    "BAR{bar_idx} mmappable area: [{:#x}..{:#x}]",
                    area.gpa,
                    area.gpa + area.size
                );
            }
        }
        if bars.bars[bar_idx as usize].is_64bit() {
            bar_idx += 1;
        }
        bar_idx += 1;
    }
    Ok((mmappable_areas, emulated_areas))
}

/// Move emulated areas and mappings located in the BAR at `from` to the BAR at `to`. Parts of the
/// BAR are found by their current address, so calling this with swapped `from` and `to` after a
/// failure moves back only the parts which were already moved.
fn vfio_move_bar(
    emulated_areas: &mut [VfioBarEmulatedArea],
    bar_mappings: &mut [VfioBarMapping],
    from: u64,
    to: u64,
    size: u64,
    vm: &KvmVm,
) -> Result<(), VfioError> {
    for mapping in bar_mappings.iter_mut() {
        if from <= mapping.gpa && mapping.gpa < from + size {
            let new_gpa = mapping.gpa - from + to;
            vm.set_user_memory_region(kvm_userspace_memory_region {
                slot: mapping.kvm_slot,
                flags: mapping.kvm_flags,
                guest_phys_addr: new_gpa,
                memory_size: mapping.size,
                userspace_addr: mapping.hva,
            })
            .map_err(VfioError::SetUserMemoryRegion)?;
            mapping.gpa = new_gpa;
        }
    }
    for area in emulated_areas.iter_mut() {
        if from <= area.gpa && area.gpa < from + size {
            let new_gpa = area.gpa - from + to;
            vm.common
                .mmio_bus
                .move_range(area.gpa, new_gpa, area.size)
                .map_err(VfioError::MoveBusRange)?;
            area.gpa = new_gpa;
        }
    }
    Ok(())
}

/// Relocate the BAR if a new address was written to its registers
fn vfio_maybe_relocate_bar(
    bars: &mut VfioBars,
    bar_idx: u8,
    emulated_areas: &mut [VfioBarEmulatedArea],
    bar_mappings: &mut [VfioBarMapping],
    vm: &KvmVm,
) {
    let old_base = bars.vmm_bars.get_bar_addr(bar_idx);
    let new_base = bars.guest_bars.get_bar_addr(bar_idx);
    if new_base == old_base {
        return;
    }

    let size = bars.vmm_bars.get_bar_size(bar_idx);
    let is_64bit = bars.vmm_bars.bars[bar_idx as usize].is_64bit();
    let mut resource_allocator = vm.resource_allocator();

    let new_base_allocator = vfio_bar_allocator(&mut resource_allocator, new_base);
    let new_range = match new_base_allocator.allocate(size, size, AllocPolicy::ExactMatch(new_base))
    {
        Ok(range) => range,
        Err(err) => {
            error!("Cannot reserve relocated BAR{bar_idx} range at {new_base:#x}: {err:?}");
            return;
        }
    };

    if let Err(err) = vfio_move_bar(emulated_areas, bar_mappings, old_base, new_base, size, vm) {
        error!("Failed to relocate BAR{bar_idx} {old_base:#x} -> {new_base:#x}: {err}");
        if let Err(err) = vfio_move_bar(emulated_areas, bar_mappings, new_base, old_base, size, vm)
        {
            // Do not return the possibly "poisoned" address range to the allocator.
            error!("Failed to move BAR{bar_idx} back to {old_base:#x}: {err}");
        } else {
            new_base_allocator.free(&new_range).unwrap();
        }
        return;
    }

    // SAFETY: the old range was allocated for this BAR
    let old_range = RangeInclusive::new(old_base, old_base + size - 1).unwrap();
    let old_base_allocator = vfio_bar_allocator(&mut resource_allocator, old_base);
    old_base_allocator.free(&old_range).unwrap();

    let i = bar_idx as usize;
    bars.vmm_bars.bars[i].encoded_addr = bars.guest_bars.bars[i].encoded_addr;
    if is_64bit {
        bars.vmm_bars.bars[i + 1].encoded_addr = bars.guest_bars.bars[i + 1].encoded_addr;
    }
    debug!("Relocated BAR{bar_idx} {old_base:#x} -> {new_base:#x}");
}

/// Relocate all BARs which have a new address written to their registers
fn vfio_maybe_relocate_all_bars(device: &mut VfioDevice) {
    let mut bar_idx = 0;
    while bar_idx < NUM_BAR_REGS {
        let bar = device.bars.vmm_bars.bars[bar_idx as usize];
        if bar.used() {
            vfio_maybe_relocate_bar(
                &mut device.bars,
                bar_idx,
                &mut device.emulated_areas,
                &mut device.bar_mappings,
                &device.vm,
            );
        }
        if bar.is_64bit() {
            bar_idx += 1;
        }
        bar_idx += 1;
    }
}

/// Mmaps the area of the device BAR and creates a sets the KVM memory region for it, giving guest
/// direct access to that memory.
fn vfio_map_bar_mapping(
    device: &impl AsRawFd,
    vm: &KvmVm,
    area: &VfioBarMappableArea,
    slot: u32,
) -> Result<VfioBarMapping, VfioError> {
    // SAFETY: FFI call to mmap with valid fd and offset. The returned pointer is checked
    // against MAP_FAILED before use.
    let hva_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            u64_to_usize(area.size),
            area.prot,
            libc::MAP_SHARED,
            device.as_raw_fd(),
            #[allow(clippy::cast_possible_wrap)]
            {
                area.vfio_fd_offset as libc::off_t
            },
        )
    };

    if hva_ptr == libc::MAP_FAILED {
        return Err(VfioError::Mmap(std::io::Error::last_os_error()));
    }

    let gpa = area.gpa;
    let size = area.size;
    let hva = hva_ptr as u64;

    let kvm_flags = if (area.prot & libc::PROT_WRITE) == 0 {
        KVM_MEM_READONLY
    } else {
        0
    };
    let kvm_memory_region = kvm_userspace_memory_region {
        slot,
        flags: kvm_flags,
        guest_phys_addr: gpa,
        memory_size: size,
        userspace_addr: hva,
    };
    if let Err(e) = vm.set_user_memory_region(kvm_memory_region) {
        // SAFETY: hva_ptr was returned by a successful mmap call above with the given size.
        let r = unsafe { libc::munmap(hva_ptr.cast(), u64_to_usize(size)) };
        if r < 0 {
            error!(
                "Error on unmapping host memory on VFIO device creation failure: {:?}. Continuing \
                 with other regions removal.",
                std::io::Error::last_os_error()
            );
        }
        return Err(VfioError::SetUserMemoryRegion(e));
    }

    // We don't establish DMA mappings for BARs yet since we don't support PTP yet. Device can
    // access their own memory without DMA.

    Ok(VfioBarMapping {
        kvm_slot: slot,
        kvm_flags,
        gpa,
        size,
        hva,
    })
}

/// Create new [`VfioBarMapping`]s from [`VfioBarMappableArea`]s
fn vfio_create_bar_mappings_from_areas(
    vm: &KvmVm,
    areas: &[VfioBarMappableArea],
    device: &impl AsRawFd,
    first_area_slot: u32,
) -> Result<Vec<VfioBarMapping>, VfioError> {
    let mut mappings = Vec::with_capacity(areas.len());
    for (i, area) in areas.iter().enumerate() {
        // `areas` length is bound by `u32`. See `vfio_calculate_bar_areas` comment.
        #[allow(clippy::cast_possible_truncation)]
        let i = i as u32;
        match vfio_map_bar_mapping(device, vm, area, first_area_slot + i) {
            Ok(mapping) => {
                debug!(
                    "BAR area{} kvm gpa: [{:#x} ..{:#x}]",
                    i,
                    mapping.gpa,
                    mapping.gpa + mapping.size
                );
                mappings.push(mapping);
            }
            Err(e) => {
                for mapping in mappings.iter() {
                    vfio_unmap_bar_mapping(vm, mapping);
                }
                return Err(e);
            }
        }
    }
    Ok(mappings)
}

/// Removes the KVM memory region and unmaps the corresponding virtual address space
fn vfio_unmap_bar_mapping(vm: &KvmVm, mapping: &VfioBarMapping) {
    let kvm_memory_region = kvm_userspace_memory_region {
        slot: mapping.kvm_slot,
        flags: 0,
        guest_phys_addr: mapping.gpa,
        memory_size: 0,
        userspace_addr: mapping.hva,
    };
    if let Err(ee) = vm.set_user_memory_region(kvm_memory_region) {
        error!(
            "Error on removing KVM region for BAR in a VFIO device: {ee:?}. Continuing with other \
             regions removal."
        );
    }

    // SAFETY: host_addr was obtained from a successful mmap call with the given size.
    let r = unsafe { libc::munmap(mapping.hva as *mut libc::c_void, u64_to_usize(mapping.size)) };
    if r < 0 {
        error!(
            "Error on unmapping host memory for BAR in a VFIO device: {:?}. Continuing with other \
             regions removal.",
            std::io::Error::last_os_error()
        );
    }
}

/// This will open a VFIO device, attach its group both to the KVM VFIO device and to the VFIO
/// container. It will setup MSIx irqs and BAR DMAs.
fn vfio_init_device(
    container: &Arc<VfioContainer>,
    vm: &Arc<KvmVm>,
    config: DevicePassthroughConfig,
    sbdf: PciSBDF,
) -> Result<VfioDevice, VfioError> {
    let sysfs_path = format!(
        "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:x}",
        config.sbdf.segment(),
        config.sbdf.bus(),
        config.sbdf.device(),
        config.sbdf.function()
    );
    debug!("Opening device at path: {}", sysfs_path);

    let device = InternalVfioDevice::new(
        Path::new(&sysfs_path),
        container.clone() as Arc<dyn vfio_ioctls::VfioOps>,
    )?;
    vfio_device_reset_command_reg(&device);
    device.reset();

    let Some(msix_irq_info) = device.get_irq_info(VFIO_PCI_MSIX_IRQ_INDEX) else {
        return Err(VfioError::NoMsixIrq);
    };
    if msix_irq_info.count == 0 {
        return Err(VfioError::NoMsixIrq);
    }

    let mut config_space = [0_u32; PCI_CONFIG_SPACE_REGS as usize];
    device.region_read(VFIO_PCI_CONFIG_REGION_INDEX, config_space.as_mut_bytes(), 0);
    let (msix_cap_and_register, masks) = vfio_get_pci_capabilities(&config_space);

    // Only devices with MSI-X cap and irqs are supported
    let Some((msix_cap, msix_register)) = msix_cap_and_register else {
        return Err(VfioError::NoMsixIrq);
    };

    // SAFETY: maximum msix table size is 1 << 11 = 2048 (it has 10 bits in the control register
    // and encoded as N - 1). This fits into u16 without issues.
    #[allow(clippy::cast_possible_truncation)]
    let msix_num = msix_irq_info.count as u16;
    let msix_vectors =
        KvmVm::create_msix_group(vm.clone(), msix_num).map_err(VfioError::MsixConfig)?;
    let msix_config = MsixConfig::new(Arc::new(msix_vectors), sbdf);

    // We set VFIO irqs here on device setup. There is no reason to add additional tracking
    // for driver MSIx configuration since those are handled by the MsixState.
    // If anything after this call fails, we don't need to do anything since the kernel will
    // clean up these irqs when `device` file will be closed.
    let fds: Vec<&EventFd> = msix_config
        .vectors
        .vectors
        .iter()
        .map(|v| &v.event_fd)
        .collect();
    device.enable_msix(fds)?;

    // There is no direct access to `regions` in `VfioDevice`, so need to work around this
    let bar_infos: [VfioBarInfo; NUM_BAR_REGS as usize] = std::array::from_fn(|i| {
        let bar_offset = PCI_CONFIG_BAR_OFFSET as usize / std::mem::size_of::<u32>() + i;
        let bar_value = config_space[bar_offset];
        #[allow(clippy::cast_possible_truncation)]
        VfioBarInfo {
            bar_value,
            flags: device.get_region_flags(i as u32),
            size: device.get_region_size(i as u32),
            offset: device.get_region_offset(i as u32),
            caps: device.get_region_caps(i as u32),
        }
    });

    let bars = VfioBars::new(vm.clone(), &bar_infos)?;

    let (areas, emulated_areas) = vfio_calculate_bar_areas(&bars.vmm_bars, &bar_infos, &msix_cap)?;
    if !emulated_areas
        .iter()
        .any(|area| area.bar_idx == msix_cap.table_bir())
    {
        return Err(VfioError::NoMsixIrq);
    }

    let Some(first_area_slot) = vm.next_kvm_slot(
        // SAFETY: areas.len() is bound to fit in u32
        #[allow(clippy::cast_possible_truncation)]
        {
            areas.len() as u32
        },
    ) else {
        return Err(VfioError::KvmSlot);
    };

    let bar_mappings =
        vfio_create_bar_mappings_from_areas(vm.as_ref(), &areas, &device, first_area_slot)?;

    let msix_state = VfioMsixState {
        register: msix_register,
        cap: msix_cap,
        config: msix_config,
    };

    let vfio_device = VfioDevice {
        config,
        sbdf,
        device,
        bars,
        bar_mappings,
        emulated_areas,
        msix_state,
        masks,
        vm: vm.clone(),
    };

    Ok(vfio_device)
}

/// Performs device teardown.
fn vfio_deinit_device(device: &VfioDevice) {
    device.device.reset();

    for mapping in device.bar_mappings.iter() {
        vfio_unmap_bar_mapping(device.vm.as_ref(), mapping);
    }
}

/// Establish DMA mapping of the DRAM regions of the guest memory with the VFIO container
/// This should be only called once after the `VfioContainer` is created.
/// If this function returns an error, the container should be destroyed.
pub fn vfio_dma_map_guest_memory(
    container: &VfioContainer,
    guest_memory: &GuestMemoryMmap,
) -> Result<(), VfioError> {
    for region in guest_memory.iter() {
        if region.region_type == GuestRegionType::Dram {
            let region = &region.inner;
            let hva = region.as_ptr();
            let iova = region.start_addr().0;
            let size = region.size();
            debug!(
                "DMA map guest memory: [{:#x}..{:#x}]",
                iova,
                iova + size as u64
            );
            // SAFETY: all arguments are from the existing guest memory region
            // After this operation, virtual memory will have pinned physical pages backing it
            unsafe { container.vfio_dma_map(iova, size, hva) }?;
        }
    }
    Ok(())
}

/// Create a VfioContainer wrapper around both KVM VFIO device and VFIO container
pub fn vfio_create_kvm_vfio_device_and_vfio_container(
    vm: &KvmVm,
) -> Result<Arc<VfioContainer>, VfioError> {
    let mut vfio_dev = kvm_create_device {
        type_: kvm_device_type_KVM_DEV_TYPE_VFIO,
        fd: 0,
        flags: 0,
    };
    let kvm_device_fd = vm
        .fd()
        .create_device(&mut vfio_dev)
        .map_err(VfioError::KvmCreateVfioDevice)?;

    let device_fd = VfioDeviceFd::new_from_kvm(kvm_device_fd);
    let container = VfioContainer::new(Some(Arc::new(device_fd)))?;
    Ok(Arc::new(container))
}
