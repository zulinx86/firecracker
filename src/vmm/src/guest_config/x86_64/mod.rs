// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/// Module for static CPU templates for x86_64
pub mod static_cpu_templates;

use std::collections::HashMap;

use super::cpuid::CpuidKey;
use super::templates::x86_64::CpuidRegister;
use super::templates::CustomCpuTemplate;
use crate::guest_config::cpuid::Cpuid;

/// Errors thrown while configuring templates.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// Failure in processing the CPUID in template for x86_64 CPU configuration.
    #[error("Template changes a CPUID entry not supported by KVM: Leaf: {0:0x}, Subleaf: {1:0x}")]
    CpuidFeatureNotSupported(u32, u32),
    /// Failure in processing the MSRs in template for x86_64 CPU configuration.
    #[error("Template changes an MSR entry not supported by KVM: Register Address: {0:0x}")]
    MsrNotSupported(u32),
    /// Can create cpuid from raw.
    #[error("Can create cpuid from raw: {0}")]
    CpuidFromKvmCpuid(super::cpuid::CpuidTryFromKvmCpuid),
    /// KVM vcpu ioctls failed.
    #[error("KVM vcpu ioctl failed: {0}")]
    VcpuIoctl(crate::vstate::vcpu::VcpuError),
}

/// CPU configuration for x86_64 CPUs
#[derive(Debug, Clone, PartialEq)]
pub struct CpuConfiguration {
    /// CPUID configuration
    pub cpuid: Cpuid,
    /// Register values as a key pair for model specific registers
    /// Key: MSR address
    /// Value: MSR value
    pub msrs: HashMap<u32, u64>,
}

impl CpuConfiguration {
    /// Modifies provided config with changes from template
    pub fn apply_template(self, template: &CustomCpuTemplate) -> Result<Self, Error> {
        let Self {
            mut cpuid,
            mut msrs,
        } = self;

        let guest_cpuid = cpuid.inner_mut();

        // Apply CPUID modifiers
        for mod_leaf in template.cpuid_modifiers.iter() {
            let cpuid_key = CpuidKey {
                leaf: mod_leaf.leaf,
                subleaf: mod_leaf.subleaf,
            };
            if let Some(entry) = guest_cpuid.get_mut(&cpuid_key) {
                entry.flags = mod_leaf.flags;

                // Can we modify one reg multiple times????
                for mod_reg in &mod_leaf.modifiers {
                    match mod_reg.register {
                        CpuidRegister::Eax => {
                            entry.result.eax =
                                mod_reg.bitmap.apply(u64::from(entry.result.eax)) as u32
                        }
                        CpuidRegister::Ebx => {
                            entry.result.ebx =
                                mod_reg.bitmap.apply(u64::from(entry.result.ebx)) as u32
                        }
                        CpuidRegister::Ecx => {
                            entry.result.ecx =
                                mod_reg.bitmap.apply(u64::from(entry.result.ecx)) as u32
                        }
                        CpuidRegister::Edx => {
                            entry.result.edx =
                                mod_reg.bitmap.apply(u64::from(entry.result.edx)) as u32
                        }
                    }
                }
            } else {
                return Err(Error::CpuidFeatureNotSupported(
                    cpuid_key.leaf,
                    cpuid_key.subleaf,
                ));
            }
        }

        for modifier in &template.msr_modifiers {
            if let Some(reg_value) = msrs.get_mut(&modifier.addr) {
                *reg_value = modifier.bitmap.apply(*reg_value);
            } else {
                return Err(Error::MsrNotSupported(modifier.addr));
            }
        }

        Ok(Self { cpuid, msrs })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use kvm_bindings::KVM_CPUID_FLAG_STATEFUL_FUNC;

    use super::*;
    use crate::guest_config::cpuid::{CpuidEntry, IntelCpuid, KvmCpuidFlags};
    use crate::guest_config::templates::x86_64::{
        CpuidLeafModifier, CpuidRegisterModifier, RegisterModifier, RegisterValueFilter,
    };

    fn build_test_template() -> CustomCpuTemplate {
        CustomCpuTemplate {
            cpuid_modifiers: vec![CpuidLeafModifier {
                leaf: 0x3,
                subleaf: 0x0,
                flags: KvmCpuidFlags(KVM_CPUID_FLAG_STATEFUL_FUNC),
                modifiers: vec![
                    CpuidRegisterModifier {
                        register: CpuidRegister::Eax,
                        bitmap: RegisterValueFilter {
                            filter: 0b0111,
                            value: 0b0101,
                        },
                    },
                    CpuidRegisterModifier {
                        register: CpuidRegister::Ebx,
                        bitmap: RegisterValueFilter {
                            filter: 0b0111,
                            value: 0b0100,
                        },
                    },
                    CpuidRegisterModifier {
                        register: CpuidRegister::Ecx,
                        bitmap: RegisterValueFilter {
                            filter: 0b0111,
                            value: 0b0111,
                        },
                    },
                    CpuidRegisterModifier {
                        register: CpuidRegister::Edx,
                        bitmap: RegisterValueFilter {
                            filter: 0b0111,
                            value: 0b0001,
                        },
                    },
                ],
            }],
            msr_modifiers: vec![
                RegisterModifier {
                    addr: 0x9999,
                    bitmap: RegisterValueFilter {
                        filter: 0,
                        value: 0,
                    },
                },
                RegisterModifier {
                    addr: 0x8000,
                    bitmap: RegisterValueFilter {
                        filter: 0,
                        value: 0,
                    },
                },
            ],
        }
    }

    fn build_supported_cpuid() -> Cpuid {
        Cpuid::Intel(IntelCpuid(BTreeMap::from([(
            CpuidKey {
                leaf: 0x3,
                subleaf: 0x0,
            },
            CpuidEntry::default(),
        )])))
    }

    fn empty_cpu_config() -> CpuConfiguration {
        CpuConfiguration {
            cpuid: Cpuid::Intel(IntelCpuid(BTreeMap::new())),
            msrs: Default::default(),
        }
    }

    fn supported_cpu_config() -> CpuConfiguration {
        CpuConfiguration {
            cpuid: build_supported_cpuid(),
            msrs: HashMap::from([(0x8000, 0b1000), (0x9999, 0b1010)]),
        }
    }

    fn unsupported_cpu_config() -> CpuConfiguration {
        CpuConfiguration {
            cpuid: build_supported_cpuid(),
            msrs: HashMap::from([(0x8000, 0b1000), (0x8001, 0b1010)]),
        }
    }

    #[test]
    fn test_empty_template() {
        let host_configuration = empty_cpu_config();
        let guest_config_result = host_configuration
            .clone()
            .apply_template(&CustomCpuTemplate::default());
        assert!(
            guest_config_result.is_ok(),
            "{}",
            guest_config_result.unwrap_err()
        );
        // CPUID will be comparable, but not MSRs.
        // The configuration will be configuration required by the template,
        // not a holistic view of all registers.
        assert_eq!(guest_config_result.unwrap().cpuid, host_configuration.cpuid);
    }

    #[test]
    fn test_apply_template() {
        let host_configuration = supported_cpu_config();
        let guest_config_result = host_configuration
            .clone()
            .apply_template(&build_test_template());
        assert!(
            guest_config_result.is_ok(),
            "{}",
            guest_config_result.unwrap_err()
        );
        assert_ne!(guest_config_result.unwrap(), host_configuration);
    }

    /// Invalid test in this context is when the template
    /// has modifiers for registers that are not supported.
    #[test]
    fn test_invalid_template() {
        // Test CPUID validation
        let host_configuration = empty_cpu_config();
        let guest_template = build_test_template();
        let guest_config_result = host_configuration.apply_template(&guest_template);
        assert!(
            guest_config_result.is_err(),
            "Expected an error as template should have failed to modify a CPUID entry that is not \
             supported by host configuration",
        );
        assert_eq!(
            guest_config_result.unwrap_err(),
            Error::CpuidFeatureNotSupported(
                guest_template.cpuid_modifiers[0].leaf,
                guest_template.cpuid_modifiers[0].subleaf
            )
        );

        // Test MSR validation
        let host_configuration = unsupported_cpu_config();
        let guest_template = build_test_template();
        let guest_config_result = host_configuration.apply_template(&guest_template);
        assert!(
            guest_config_result.is_err(),
            "Expected an error as template should have failed to modify an MSR value that is not \
             supported by host configuration",
        );
        assert_eq!(
            guest_config_result.unwrap_err(),
            Error::MsrNotSupported(guest_template.msr_modifiers[0].addr)
        )
    }
}