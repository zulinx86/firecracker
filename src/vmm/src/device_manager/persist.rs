// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provides functionality for saving/restoring the MMIO device manager and its devices.

use std::fmt::{self, Debug};
use std::sync::{Arc, Mutex};

use event_manager::{MutEventSubscriber, SubscriberOps};
use kvm_ioctls::VmFd;
use log::{error, warn};
use serde::{Deserialize, Serialize};
use vm_allocator::AllocPolicy;

use super::acpi::ACPIDeviceManager;
use super::mmio::*;
use super::resources::ResourceAllocator;
use crate::EventManager;
#[cfg(target_arch = "aarch64")]
use crate::arch::DeviceType;
use crate::devices::acpi::vmgenid::{VMGenIDState, VMGenIdConstructorArgs, VmGenId, VmGenIdError};
use crate::devices::virtio::balloon::persist::{BalloonConstructorArgs, BalloonState};
use crate::devices::virtio::balloon::{Balloon, BalloonError};
use crate::devices::virtio::block::BlockError;
use crate::devices::virtio::block::device::Block;
use crate::devices::virtio::block::persist::{BlockConstructorArgs, BlockState};
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::mmio::MmioTransport;
use crate::devices::virtio::net::Net;
use crate::devices::virtio::net::persist::{
    NetConstructorArgs, NetPersistError as NetError, NetState,
};
use crate::devices::virtio::persist::{MmioTransportConstructorArgs, MmioTransportState};
use crate::devices::virtio::rng::Entropy;
use crate::devices::virtio::rng::persist::{
    EntropyConstructorArgs, EntropyPersistError as EntropyError, EntropyState,
};
use crate::devices::virtio::vsock::persist::{
    VsockConstructorArgs, VsockState, VsockUdsConstructorArgs,
};
use crate::devices::virtio::vsock::{
    TYPE_VSOCK, Vsock, VsockError, VsockUnixBackend, VsockUnixBackendError,
};
use crate::devices::virtio::{TYPE_BALLOON, TYPE_BLOCK, TYPE_NET, TYPE_RNG};
use crate::mmds::data_store::MmdsVersion;
use crate::resources::{ResourcesError, VmResources};
use crate::snapshot::Persist;
use crate::vmm_config::mmds::MmdsConfigError;
use crate::vstate::memory::GuestMemoryMmap;

/// Errors for (de)serialization of the MMIO device manager.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DevicePersistError {
    /// Balloon: {0}
    Balloon(#[from] BalloonError),
    /// Block: {0}
    Block(#[from] BlockError),
    /// Device manager: {0}
    DeviceManager(#[from] super::mmio::MmioError),
    /// Mmio transport
    MmioTransport,
    #[cfg(target_arch = "aarch64")]
    /// Legacy: {0}
    Legacy(#[from] crate::VmmError),
    /// Net: {0}
    Net(#[from] NetError),
    /// Vsock: {0}
    Vsock(#[from] VsockError),
    /// VsockUnixBackend: {0}
    VsockUnixBackend(#[from] VsockUnixBackendError),
    /// MmdsConfig: {0}
    MmdsConfig(#[from] MmdsConfigError),
    /// Entropy: {0}
    Entropy(#[from] EntropyError),
    /// Resource misconfiguration: {0}. Is the snapshot file corrupted?
    ResourcesError(#[from] ResourcesError),
}

/// Holds the state of a balloon device connected to the MMIO space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedBalloonState {
    /// Device identifier.
    pub device_id: String,
    /// Device state.
    pub device_state: BalloonState,
    /// Mmio transport state.
    pub transport_state: MmioTransportState,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

/// Holds the state of a virtio block device connected to the MMIO space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedBlockState {
    /// Device identifier.
    pub device_id: String,
    /// Device state.
    pub device_state: BlockState,
    /// Mmio transport state.
    pub transport_state: MmioTransportState,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

/// Holds the state of a net device connected to the MMIO space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedNetState {
    /// Device identifier.
    pub device_id: String,
    /// Device state.
    pub device_state: NetState,
    /// Mmio transport state.
    pub transport_state: MmioTransportState,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

/// Holds the state of a vsock device connected to the MMIO space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedVsockState {
    /// Device identifier.
    pub device_id: String,
    /// Device state.
    pub device_state: VsockState,
    /// Mmio transport state.
    pub transport_state: MmioTransportState,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

/// Holds the state of an entropy device connected to the MMIO space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedEntropyState {
    /// Device identifier.
    pub device_id: String,
    /// Device state.
    pub device_state: EntropyState,
    /// Mmio transport state.
    pub transport_state: MmioTransportState,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

/// Holds the state of a legacy device connected to the MMIO space.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedLegacyState {
    /// Device identifier.
    pub type_: DeviceType,
    /// VmmResources.
    pub device_info: MMIODeviceInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmdsState {
    version: MmdsVersion,
    imds_compat: bool,
}

/// Holds the device states.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DeviceStates {
    #[cfg(target_arch = "aarch64")]
    // State of legacy devices in MMIO space.
    pub legacy_devices: Vec<ConnectedLegacyState>,
    /// Block device states.
    pub block_devices: Vec<ConnectedBlockState>,
    /// Net device states.
    pub net_devices: Vec<ConnectedNetState>,
    /// Vsock device state.
    pub vsock_device: Option<ConnectedVsockState>,
    /// Balloon device state.
    pub balloon_device: Option<ConnectedBalloonState>,
    /// Mmds version.
    pub mmds: Option<MmdsState>,
    /// Entropy device state.
    pub entropy_device: Option<ConnectedEntropyState>,
}

/// A type used to extract the concrete `Arc<Mutex<T>>` for each of the device
/// types when restoring from a snapshot.
#[derive(Debug)]
pub enum SharedDeviceType {
    VirtioBlock(Arc<Mutex<Block>>),
    Network(Arc<Mutex<Net>>),
    Balloon(Arc<Mutex<Balloon>>),
    Vsock(Arc<Mutex<Vsock<VsockUnixBackend>>>),
    Entropy(Arc<Mutex<Entropy>>),
}

pub struct MMIODevManagerConstructorArgs<'a> {
    pub mem: &'a GuestMemoryMmap,
    pub vm: &'a VmFd,
    pub event_manager: &'a mut EventManager,
    pub resource_allocator: &'a mut ResourceAllocator,
    pub vm_resources: &'a mut VmResources,
    pub instance_id: &'a str,
    pub restored_from_file: bool,
}
impl fmt::Debug for MMIODevManagerConstructorArgs<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MMIODevManagerConstructorArgs")
            .field("mem", &self.mem)
            .field("vm", &self.vm)
            .field("event_manager", &"?")
            .field("for_each_restored_device", &"?")
            .field("vm_resources", &self.vm_resources)
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ACPIDeviceManagerState {
    vmgenid: Option<VMGenIDState>,
}

pub struct ACPIDeviceManagerConstructorArgs<'a> {
    pub mem: &'a GuestMemoryMmap,
    pub resource_allocator: &'a mut ResourceAllocator,
    pub vm: &'a VmFd,
}

#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum ACPIDeviceManagerRestoreError {
    /// Could not register device: {0}
    Interrupt(#[from] kvm_ioctls::Error),
    /// Could not create VMGenID device: {0}
    VMGenID(#[from] VmGenIdError),
}

impl<'a> Persist<'a> for ACPIDeviceManager {
    type State = ACPIDeviceManagerState;
    type ConstructorArgs = ACPIDeviceManagerConstructorArgs<'a>;
    type Error = ACPIDeviceManagerRestoreError;

    fn save(&self) -> Self::State {
        ACPIDeviceManagerState {
            vmgenid: self.vmgenid.as_ref().map(|dev| dev.save()),
        }
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> std::result::Result<Self, Self::Error> {
        let mut dev_manager = ACPIDeviceManager::new();
        if let Some(vmgenid_args) = &state.vmgenid {
            let vmgenid = VmGenId::restore(
                VMGenIdConstructorArgs {
                    mem: constructor_args.mem,
                    resource_allocator: constructor_args.resource_allocator,
                },
                vmgenid_args,
            )?;
            dev_manager.attach_vmgenid(vmgenid, constructor_args.vm)?;
        }
        Ok(dev_manager)
    }
}

impl<'a> Persist<'a> for MMIODeviceManager {
    type State = DeviceStates;
    type ConstructorArgs = MMIODevManagerConstructorArgs<'a>;
    type Error = DevicePersistError;

    fn save(&self) -> Self::State {
        let mut states = DeviceStates::default();
        let _: Result<(), ()> = self.for_each_device(|devtype, devid, device_info, bus_dev| {
            if *devtype == crate::arch::DeviceType::BootTimer {
                // No need to save BootTimer state.
                return Ok(());
            }

            #[cfg(target_arch = "aarch64")]
            {
                if *devtype == DeviceType::Serial || *devtype == DeviceType::Rtc {
                    states.legacy_devices.push(ConnectedLegacyState {
                        type_: *devtype,
                        device_info: device_info.clone(),
                    });
                    return Ok(());
                }
            }

            let locked_bus_dev = bus_dev.lock().expect("Poisoned lock");

            let mmio_transport = locked_bus_dev
                .mmio_transport_ref()
                .expect("Unexpected device type");

            let transport_state = mmio_transport.save();

            let mut locked_device = mmio_transport.locked_device();
            match locked_device.device_type() {
                TYPE_BALLOON => {
                    let balloon_state = locked_device
                        .as_any()
                        .downcast_ref::<Balloon>()
                        .unwrap()
                        .save();
                    states.balloon_device = Some(ConnectedBalloonState {
                        device_id: devid.clone(),
                        device_state: balloon_state,
                        transport_state,
                        device_info: device_info.clone(),
                    });
                }
                // Both virtio-block and vhost-user-block share same device type.
                TYPE_BLOCK => {
                    let block = locked_device.as_mut_any().downcast_mut::<Block>().unwrap();
                    if block.is_vhost_user() {
                        warn!(
                            "Skipping vhost-user-block device. VhostUserBlock does not support \
                             snapshotting yet"
                        );
                    } else {
                        block.prepare_save();
                        states.block_devices.push(ConnectedBlockState {
                            device_id: devid.clone(),
                            device_state: block.save(),
                            transport_state,
                            device_info: device_info.clone(),
                        })
                    }
                }
                TYPE_NET => {
                    let net = locked_device.as_any().downcast_ref::<Net>().unwrap();
                    if let (Some(mmds_ns), None) = (net.mmds_ns.as_ref(), states.mmds.as_ref()) {
                        let mmds_guard = mmds_ns.mmds.lock().expect("Poisoned lock");
                        states.mmds = Some(MmdsState {
                            version: mmds_guard.version(),
                            imds_compat: mmds_guard.imds_compat(),
                        });
                    }

                    states.net_devices.push(ConnectedNetState {
                        device_id: devid.clone(),
                        device_state: net.save(),
                        transport_state,
                        device_info: device_info.clone(),
                    });
                }
                TYPE_VSOCK => {
                    let vsock = locked_device
                        .as_mut_any()
                        // Currently, VsockUnixBackend is the only implementation of VsockBackend.
                        .downcast_mut::<Vsock<VsockUnixBackend>>()
                        .unwrap();

                    // Send Transport event to reset connections if device
                    // is activated.
                    if vsock.is_activated() {
                        vsock.send_transport_reset_event().unwrap_or_else(|err| {
                            error!("Failed to send reset transport event: {:?}", err);
                        });
                    }

                    // Save state after potential notification to the guest. This
                    // way we save changes to the queue the notification can cause.
                    let vsock_state = VsockState {
                        backend: vsock.backend().save(),
                        frontend: vsock.save(),
                    };

                    states.vsock_device = Some(ConnectedVsockState {
                        device_id: devid.clone(),
                        device_state: vsock_state,
                        transport_state,
                        device_info: device_info.clone(),
                    });
                }
                TYPE_RNG => {
                    let entropy = locked_device
                        .as_mut_any()
                        .downcast_mut::<Entropy>()
                        .unwrap();

                    states.entropy_device = Some(ConnectedEntropyState {
                        device_id: devid.clone(),
                        device_state: entropy.save(),
                        transport_state,
                        device_info: device_info.clone(),
                    });
                }
                _ => unreachable!(),
            };

            Ok(())
        });
        states
    }

    fn restore(
        constructor_args: Self::ConstructorArgs,
        state: &Self::State,
    ) -> Result<Self, Self::Error> {
        let mut dev_manager = MMIODeviceManager::new();
        let mem = constructor_args.mem;
        let vm = constructor_args.vm;

        #[cfg(target_arch = "aarch64")]
        {
            for state in &state.legacy_devices {
                if state.type_ == DeviceType::Serial {
                    let serial = crate::builder::setup_serial_device(
                        constructor_args.event_manager,
                        std::io::stdin(),
                        std::io::stdout(),
                    )?;

                    constructor_args
                        .resource_allocator
                        .allocate_mmio_memory(
                            MMIO_LEN,
                            MMIO_LEN,
                            AllocPolicy::ExactMatch(state.device_info.addr),
                        )
                        .map_err(|e| {
                            DevicePersistError::DeviceManager(super::mmio::MmioError::Allocator(e))
                        })?;

                    dev_manager.register_mmio_serial(
                        vm,
                        constructor_args.resource_allocator,
                        serial,
                        Some(state.device_info.clone()),
                    )?;
                }
                if state.type_ == DeviceType::Rtc {
                    let rtc = crate::devices::legacy::RTCDevice(vm_superio::Rtc::with_events(
                        &crate::devices::legacy::rtc_pl031::METRICS,
                    ));
                    constructor_args
                        .resource_allocator
                        .allocate_mmio_memory(
                            MMIO_LEN,
                            MMIO_LEN,
                            AllocPolicy::ExactMatch(state.device_info.addr),
                        )
                        .map_err(|e| {
                            DevicePersistError::DeviceManager(super::mmio::MmioError::Allocator(e))
                        })?;
                    dev_manager.register_mmio_rtc(
                        constructor_args.resource_allocator,
                        rtc,
                        Some(state.device_info.clone()),
                    )?;
                }
            }
        }

        let mut restore_helper = |device: Arc<Mutex<dyn VirtioDevice>>,
                                  is_vhost_user: bool,
                                  as_subscriber: Arc<Mutex<dyn MutEventSubscriber>>,
                                  id: &String,
                                  state: &MmioTransportState,
                                  device_info: &MMIODeviceInfo,
                                  event_manager: &mut EventManager|
         -> Result<(), Self::Error> {
            let restore_args = MmioTransportConstructorArgs {
                mem: mem.clone(),
                device,
                is_vhost_user,
            };
            let mmio_transport = MmioTransport::restore(restore_args, state)
                .map_err(|()| DevicePersistError::MmioTransport)?;

            // We do not currently require exact re-allocation of IDs via
            // `dev_manager.irq_allocator.allocate_id()` and currently cannot do
            // this effectively as `IdAllocator` does not implement an exact
            // match API.
            // In the future we may require preserving `IdAllocator`'s state
            // after snapshot restore so as to restore the exact interrupt IDs
            // from the original device's state for implementing hot-plug.
            // For now this is why we do not restore the state of the
            // `IdAllocator` under `dev_manager`.

            constructor_args
                .resource_allocator
                .allocate_mmio_memory(
                    MMIO_LEN,
                    MMIO_LEN,
                    AllocPolicy::ExactMatch(device_info.addr),
                )
                .map_err(|e| {
                    DevicePersistError::DeviceManager(super::mmio::MmioError::Allocator(e))
                })?;

            dev_manager.register_mmio_virtio(vm, id.clone(), mmio_transport, device_info)?;

            event_manager.add_subscriber(as_subscriber);
            Ok(())
        };

        if let Some(balloon_state) = &state.balloon_device {
            let device = Arc::new(Mutex::new(Balloon::restore(
                BalloonConstructorArgs {
                    mem: mem.clone(),
                    restored_from_file: constructor_args.restored_from_file,
                },
                &balloon_state.device_state,
            )?));

            constructor_args
                .vm_resources
                .update_from_restored_device(SharedDeviceType::Balloon(device.clone()))?;

            restore_helper(
                device.clone(),
                false,
                device,
                &balloon_state.device_id,
                &balloon_state.transport_state,
                &balloon_state.device_info,
                constructor_args.event_manager,
            )?;
        }

        for block_state in &state.block_devices {
            let device = Arc::new(Mutex::new(Block::restore(
                BlockConstructorArgs { mem: mem.clone() },
                &block_state.device_state,
            )?));

            constructor_args
                .vm_resources
                .update_from_restored_device(SharedDeviceType::VirtioBlock(device.clone()))?;

            restore_helper(
                device.clone(),
                false,
                device,
                &block_state.device_id,
                &block_state.transport_state,
                &block_state.device_info,
                constructor_args.event_manager,
            )?;
        }

        // Initialize MMDS if MMDS state is included.
        if let Some(mmds) = &state.mmds {
            constructor_args.vm_resources.set_mmds_basic_config(
                mmds.version,
                mmds.imds_compat,
                constructor_args.instance_id,
            )?;
        }

        for net_state in &state.net_devices {
            let device = Arc::new(Mutex::new(Net::restore(
                NetConstructorArgs {
                    mem: mem.clone(),
                    mmds: constructor_args
                        .vm_resources
                        .mmds
                        .as_ref()
                        // Clone the Arc reference.
                        .cloned(),
                },
                &net_state.device_state,
            )?));

            constructor_args
                .vm_resources
                .update_from_restored_device(SharedDeviceType::Network(device.clone()))?;

            restore_helper(
                device.clone(),
                false,
                device,
                &net_state.device_id,
                &net_state.transport_state,
                &net_state.device_info,
                constructor_args.event_manager,
            )?;
        }

        if let Some(vsock_state) = &state.vsock_device {
            let ctor_args = VsockUdsConstructorArgs {
                cid: vsock_state.device_state.frontend.cid,
            };
            let backend = VsockUnixBackend::restore(ctor_args, &vsock_state.device_state.backend)?;
            let device = Arc::new(Mutex::new(Vsock::restore(
                VsockConstructorArgs {
                    mem: mem.clone(),
                    backend,
                },
                &vsock_state.device_state.frontend,
            )?));

            constructor_args
                .vm_resources
                .update_from_restored_device(SharedDeviceType::Vsock(device.clone()))?;

            restore_helper(
                device.clone(),
                false,
                device,
                &vsock_state.device_id,
                &vsock_state.transport_state,
                &vsock_state.device_info,
                constructor_args.event_manager,
            )?;
        }

        if let Some(entropy_state) = &state.entropy_device {
            let ctor_args = EntropyConstructorArgs::new(mem.clone());

            let device = Arc::new(Mutex::new(Entropy::restore(
                ctor_args,
                &entropy_state.device_state,
            )?));

            constructor_args
                .vm_resources
                .update_from_restored_device(SharedDeviceType::Entropy(device.clone()))?;

            restore_helper(
                device.clone(),
                false,
                device,
                &entropy_state.device_id,
                &entropy_state.transport_state,
                &entropy_state.device_info,
                constructor_args.event_manager,
            )?;
        }

        Ok(dev_manager)
    }
}

#[cfg(test)]
mod tests {
    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::builder::tests::*;
    use crate::devices::virtio::block::CacheType;
    use crate::resources::VmmConfig;
    use crate::snapshot::Snapshot;
    use crate::vmm_config::balloon::BalloonDeviceConfig;
    use crate::vmm_config::entropy::EntropyDeviceConfig;
    use crate::vmm_config::net::NetworkInterfaceConfig;
    use crate::vmm_config::vsock::VsockDeviceConfig;

    impl PartialEq for ConnectedBalloonState {
        fn eq(&self, other: &ConnectedBalloonState) -> bool {
            // Actual device state equality is checked by the device's tests.
            self.transport_state == other.transport_state && self.device_info == other.device_info
        }
    }

    impl PartialEq for ConnectedBlockState {
        fn eq(&self, other: &ConnectedBlockState) -> bool {
            // Actual device state equality is checked by the device's tests.
            self.transport_state == other.transport_state && self.device_info == other.device_info
        }
    }

    impl PartialEq for ConnectedNetState {
        fn eq(&self, other: &ConnectedNetState) -> bool {
            // Actual device state equality is checked by the device's tests.
            self.transport_state == other.transport_state && self.device_info == other.device_info
        }
    }

    impl PartialEq for ConnectedVsockState {
        fn eq(&self, other: &ConnectedVsockState) -> bool {
            // Actual device state equality is checked by the device's tests.
            self.transport_state == other.transport_state && self.device_info == other.device_info
        }
    }

    impl PartialEq for DeviceStates {
        fn eq(&self, other: &DeviceStates) -> bool {
            self.balloon_device == other.balloon_device
                && self.block_devices == other.block_devices
                && self.net_devices == other.net_devices
                && self.vsock_device == other.vsock_device
        }
    }

    impl MMIODeviceManager {
        fn soft_clone(&self) -> Self {
            // We can unwrap here as we create with values directly in scope we
            // know will results in `Ok`
            let mut clone = MMIODeviceManager::new();
            // We only care about the device hashmap.
            clone.id_to_dev_info.clone_from(&self.id_to_dev_info);
            clone
        }
    }

    impl PartialEq for MMIODeviceManager {
        fn eq(&self, other: &MMIODeviceManager) -> bool {
            // We only care about the device hashmap.
            if self.id_to_dev_info.len() != other.id_to_dev_info.len() {
                return false;
            }
            for (key, val) in &self.id_to_dev_info {
                match other.id_to_dev_info.get(key) {
                    Some(other_val) if val == other_val => continue,
                    _ => return false,
                };
            }
            true
        }
    }

    #[test]
    fn test_device_manager_persistence() {
        let mut buf = vec![0; 16384];
        // These need to survive so the restored blocks find them.
        let _block_files;
        let mut tmp_sock_file = TempFile::new().unwrap();
        let mut resource_allocator = ResourceAllocator::new().unwrap();
        tmp_sock_file.remove().unwrap();
        // Set up a vmm with one of each device, and get the serialized DeviceStates.
        let original_mmio_device_manager = {
            let mut event_manager = EventManager::new().expect("Unable to create EventManager");
            let mut vmm = default_vmm();
            let mut cmdline = default_kernel_cmdline();

            // Add a balloon device.
            let balloon_cfg = BalloonDeviceConfig {
                amount_mib: 123,
                deflate_on_oom: false,
                stats_polling_interval_s: 1,
            };
            insert_balloon_device(&mut vmm, &mut cmdline, &mut event_manager, balloon_cfg);
            // Add a block device.
            let drive_id = String::from("root");
            let block_configs = vec![CustomBlockConfig::new(
                drive_id,
                true,
                None,
                true,
                CacheType::Unsafe,
            )];
            _block_files =
                insert_block_devices(&mut vmm, &mut cmdline, &mut event_manager, block_configs);
            // Add a net device.
            let network_interface = NetworkInterfaceConfig {
                iface_id: String::from("netif"),
                host_dev_name: String::from("hostname"),
                guest_mac: None,
                rx_rate_limiter: None,
                tx_rate_limiter: None,
            };
            insert_net_device_with_mmds(
                &mut vmm,
                &mut cmdline,
                &mut event_manager,
                network_interface,
                MmdsVersion::V2,
            );
            // Add a vsock device.
            let vsock_dev_id = "vsock";
            let vsock_config = VsockDeviceConfig {
                vsock_id: Some(vsock_dev_id.to_string()),
                guest_cid: 3,
                uds_path: tmp_sock_file.as_path().to_str().unwrap().to_string(),
            };
            insert_vsock_device(&mut vmm, &mut cmdline, &mut event_manager, vsock_config);
            // Add an entropy device.
            let entropy_config = EntropyDeviceConfig::default();
            insert_entropy_device(&mut vmm, &mut cmdline, &mut event_manager, entropy_config);

            Snapshot::serialize(&mut buf.as_mut_slice(), &vmm.mmio_device_manager.save()).unwrap();

            // We only want to keep the device map from the original MmioDeviceManager.
            vmm.mmio_device_manager.soft_clone()
        };
        tmp_sock_file.remove().unwrap();

        let mut event_manager = EventManager::new().expect("Unable to create EventManager");
        let vmm = default_vmm();
        let device_states: DeviceStates = Snapshot::deserialize(&mut buf.as_slice()).unwrap();
        let vm_resources = &mut VmResources::default();
        let restore_args = MMIODevManagerConstructorArgs {
            mem: vmm.vm.guest_memory(),
            vm: vmm.vm.fd(),
            event_manager: &mut event_manager,
            resource_allocator: &mut resource_allocator,
            vm_resources,
            instance_id: "microvm-id",
            restored_from_file: true,
        };
        let restored_dev_manager =
            MMIODeviceManager::restore(restore_args, &device_states).unwrap();

        let expected_vm_resources = format!(
            r#"{{
  "balloon": {{
    "amount_mib": 123,
    "deflate_on_oom": false,
    "stats_polling_interval_s": 1
  }},
  "drives": [
    {{
      "drive_id": "root",
      "partuuid": null,
      "is_root_device": true,
      "cache_type": "Unsafe",
      "is_read_only": true,
      "path_on_host": "{}",
      "rate_limiter": null,
      "io_engine": "Sync",
      "socket": null
    }}
  ],
  "boot-source": {{
    "kernel_image_path": "",
    "initrd_path": null,
    "boot_args": null
  }},
  "cpu-config": null,
  "logger": null,
  "machine-config": {{
    "vcpu_count": 1,
    "mem_size_mib": 128,
    "smt": false,
    "track_dirty_pages": false,
    "huge_pages": "None"
  }},
  "metrics": null,
  "mmds-config": {{
    "version": "V2",
    "network_interfaces": [
      "netif"
    ],
    "ipv4_address": "169.254.169.254",
    "imds_compat": false
  }},
  "network-interfaces": [
    {{
      "iface_id": "netif",
      "host_dev_name": "hostname",
      "guest_mac": null,
      "rx_rate_limiter": null,
      "tx_rate_limiter": null
    }}
  ],
  "vsock": {{
    "guest_cid": 3,
    "uds_path": "{}"
  }},
  "entropy": {{
    "rate_limiter": null
  }}
}}"#,
            _block_files.last().unwrap().as_path().to_str().unwrap(),
            tmp_sock_file.as_path().to_str().unwrap()
        );

        assert_eq!(
            vm_resources
                .mmds
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .version(),
            MmdsVersion::V2
        );
        assert_eq!(device_states.mmds.unwrap().version, MmdsVersion::V2);

        assert_eq!(restored_dev_manager, original_mmio_device_manager);
        assert_eq!(
            expected_vm_resources,
            serde_json::to_string_pretty(&VmmConfig::from(&*vm_resources)).unwrap()
        );
    }
}
