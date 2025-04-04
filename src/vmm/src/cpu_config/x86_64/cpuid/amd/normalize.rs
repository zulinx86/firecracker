// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cpu_config::x86_64::cpuid::common::{GetCpuidError, get_vendor_id_from_host};
use crate::cpu_config::x86_64::cpuid::normalize::{
    CheckedAssignError, get_range, set_bit, set_range,
};
use crate::cpu_config::x86_64::cpuid::{
    BRAND_STRING_LENGTH, CpuidEntry, CpuidKey, CpuidRegisters, CpuidTrait, KvmCpuidFlags,
    MissingBrandStringLeaves, VENDOR_ID_AMD, cpuid, cpuid_count,
};

/// Error type for [`super::AmdCpuid::normalize`].
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, thiserror::Error, displaydoc::Display, Eq, PartialEq)]
pub enum NormalizeCpuidError {
    /// Provided `cpu_bits` is >=8: {0}.
    CpuBits(u8),
    /// Failed to passthrough cache topology: {0}
    PassthroughCacheTopology(#[from] PassthroughCacheTopologyError),
    /// Missing leaf 0x7 / subleaf 0.
    MissingLeaf0x7Subleaf0,
    /// Missing leaf 0x80000000.
    MissingLeaf0x80000000,
    /// Missing leaf 0x80000001.
    MissingLeaf0x80000001,
    /// Failed to set feature entry leaf: {0}
    FeatureEntry(#[from] FeatureEntryError),
    /// Failed to set extended cache topology leaf: {0}
    ExtendedCacheTopology(#[from] ExtendedCacheTopologyError),
    /// Failed to set extended APIC ID leaf: {0}
    ExtendedApicId(#[from] ExtendedApicIdError),
    /// Failed to set brand string: {0}
    BrandString(MissingBrandStringLeaves),
}

/// Error type for setting cache topology section of [`super::AmdCpuid::normalize`].
#[derive(Debug, thiserror::Error, displaydoc::Display, Eq, PartialEq)]
pub enum PassthroughCacheTopologyError {
    /// Failed to get the host vendor id: {0}
    NoVendorId(GetCpuidError),
    /// The host vendor id does not match AMD.
    BadVendorId,
}

/// Error type for setting leaf 0x80000008 section of [`super::AmdCpuid::normalize`].
#[derive(Debug, thiserror::Error, displaydoc::Display, Eq, PartialEq)]
pub enum FeatureEntryError {
    /// Missing leaf 0x80000008.
    MissingLeaf0x80000008,
    /// Failed to set number of physical threads (CPUID.80000008H:ECX[7:0]): {0}
    NumberOfPhysicalThreads(CheckedAssignError),
    /// Failed to set number of physical threads (CPUID.80000008H:ECX[7:0]) due to overflow.
    NumberOfPhysicalThreadsOverflow,
}

/// Error type for setting leaf 0x8000001d section of [`super::AmdCpuid::normalize`].
#[derive(Debug, thiserror::Error, displaydoc::Display, Eq, PartialEq)]
pub enum ExtendedCacheTopologyError {
    /// Missing leaf 0x8000001d.
    MissingLeaf0x8000001d,
    #[rustfmt::skip]
    /// Failed to set number of logical processors sharing cache(CPUID.(EAX=8000001DH,ECX={0}):EAX[25:14]): {1}
    NumSharingCache(u32, CheckedAssignError),
    #[rustfmt::skip]
    /// Failed to set number of logical processors sharing cache (CPUID.(EAX=8000001DH,ECX={0}):EAX[25:14]) due to overflow.
    NumSharingCacheOverflow(u32),
}

/// Error type for setting leaf 0x8000001e section of [`super::AmdCpuid::normalize`].
#[derive(Debug, thiserror::Error, displaydoc::Display, Eq, PartialEq)]
pub enum ExtendedApicIdError {
    /// Failed to set compute unit ID (CPUID.8000001EH:EBX[7:0]): {0}
    ComputeUnitId(CheckedAssignError),
    /// Failed to set extended APIC ID (CPUID.8000001EH:EAX[31:0]): {0}
    ExtendedApicId(CheckedAssignError),
    /// Missing leaf 0x8000001e.
    MissingLeaf0x8000001e,
    /// Failed to set threads per core unit (CPUID:8000001EH:EBX[15:8]): {0}
    ThreadPerComputeUnit(CheckedAssignError),
}

// We use this 2nd implementation so we can conveniently define functions only used within
// `normalize`.
#[allow(clippy::multiple_inherent_impl)]
impl super::AmdCpuid {
    /// We always use this brand string.
    const DEFAULT_BRAND_STRING: &'static [u8; BRAND_STRING_LENGTH] =
        b"AMD EPYC\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

    /// Applies required modifications to CPUID respective of a vCPU.
    ///
    /// # Errors
    ///
    /// When attempting to access missing leaves or set fields within leaves to values that don't
    /// fit.
    #[inline]
    pub fn normalize(
        &mut self,
        // The index of the current logical CPU in the range [0..cpu_count].
        cpu_index: u8,
        // The total number of logical CPUs.
        cpu_count: u8,
        // The number of logical CPUs per core.
        cpus_per_core: u8,
    ) -> Result<(), NormalizeCpuidError> {
        self.passthrough_cache_topology()?;
        self.update_structured_extended_entry()?;
        self.update_extended_feature_fn_entry()?;
        self.update_amd_feature_entry(cpu_count)?;
        self.update_extended_cache_topology_entry(cpu_count, cpus_per_core)?;
        self.update_extended_apic_id_entry(cpu_index, cpus_per_core)?;
        self.update_brand_string_entry()?;

        Ok(())
    }

    /// Passthrough cache topology.
    ///
    /// # Errors
    ///
    /// This function passes through leaves from the host CPUID, if this does not match the AMD
    /// specification it is possible to enter an indefinite loop. To avoid this, this will return an
    /// error when the host CPUID vendor id does not match the AMD CPUID vendor id.
    fn passthrough_cache_topology(&mut self) -> Result<(), PassthroughCacheTopologyError> {
        if get_vendor_id_from_host().map_err(PassthroughCacheTopologyError::NoVendorId)?
            != *VENDOR_ID_AMD
        {
            return Err(PassthroughCacheTopologyError::BadVendorId);
        }

        // Pass-through host CPUID for leaves 0x8000001e and 0x8000001d.
        {
            // 0x8000001e - Processor Topology Information
            self.0.insert(
                CpuidKey::leaf(0x8000001e),
                CpuidEntry {
                    flags: KvmCpuidFlags::EMPTY,
                    result: CpuidRegisters::from(cpuid(0x8000001e)),
                },
            );

            // 0x8000001d - Cache Topology Information
            for subleaf in 0.. {
                let result = CpuidRegisters::from(cpuid_count(0x8000001d, subleaf));
                // From 'AMD64 Architecture Programmer’s Manual Volume 3: General-Purpose and System
                // Instructions':
                //
                // > To gather information for all cache levels, software must repeatedly execute
                // > CPUID with 8000_001Dh in EAX and ECX set to increasing values beginning with 0
                // > until a value of 00h is returned in the field CacheType (EAX[4:0]) indicating
                // > no more cache descriptions are available for this processor. If CPUID
                // > Fn8000_0001_ECX[TopologyExtensions] = 0, then CPUID Fn8000_001Dh is reserved.
                //
                // On non-AMD hosts this condition may never be true thus this loop may be
                // indefinite.

                // CPUID Fn8000_0001D_EAX_x[4:0] (Field Name: CacheType)
                // Cache type. Identifies the type of cache.
                // ```text
                // Bits Description
                // 00h Null; no more caches.
                // 01h Data cache
                // 02h Instruction cache
                // 03h Unified cache
                // 1Fh-04h Reserved.
                // ```
                let cache_type = result.eax & 15;
                if cache_type == 0 {
                    break;
                }
                self.0.insert(
                    CpuidKey::subleaf(0x8000001d, subleaf),
                    CpuidEntry {
                        flags: KvmCpuidFlags::SIGNIFICANT_INDEX,
                        result,
                    },
                );
            }
        }
        Ok(())
    }

    /// Updated extended feature fn entry.
    fn update_extended_feature_fn_entry(&mut self) -> Result<(), NormalizeCpuidError> {
        // set the Topology Extension bit since we use the Extended Cache Topology leaf
        let leaf_80000001 = self
            .get_mut(&CpuidKey::leaf(0x80000001))
            .ok_or(NormalizeCpuidError::MissingLeaf0x80000001)?;
        // CPUID Fn8000_0001_ECX[22] (Field Name: TopologyExtensions)
        // Topology extensions support. Indicates support for CPUID Fn8000_001D_EAX_x[N:0]-CPUID
        // Fn8000_001E_EDX.
        set_bit(&mut leaf_80000001.result.ecx, 22, true);
        Ok(())
    }

    // Update structured extended feature entry.
    fn update_structured_extended_entry(&mut self) -> Result<(), NormalizeCpuidError> {
        let leaf_7_subleaf_0 = self
            .get_mut(&CpuidKey::subleaf(0x7, 0x0))
            .ok_or(NormalizeCpuidError::MissingLeaf0x7Subleaf0)?;

        // According to AMD64 Architecture Programmer’s Manual, IA32_ARCH_CAPABILITIES MSR is not
        // available on AMD. The availability of IA32_ARCH_CAPABILITIES MSR is controlled via
        // CPUID.07H(ECX=0):EDX[bit 29]. KVM sets this bit no matter what but this feature is not
        // supported by hardware.
        set_bit(&mut leaf_7_subleaf_0.result.edx, 29, false);
        Ok(())
    }

    /// Update AMD feature entry.
    #[allow(clippy::unwrap_used, clippy::unwrap_in_result)]
    fn update_amd_feature_entry(&mut self, cpu_count: u8) -> Result<(), FeatureEntryError> {
        /// This value allows at most 64 logical threads within a package.
        const THREAD_ID_MAX_SIZE: u32 = 7;

        // We don't support more then 128 threads right now.
        // It's safe to put them all on the same processor.
        let leaf_80000008 = self
            .get_mut(&CpuidKey::leaf(0x80000008))
            .ok_or(FeatureEntryError::MissingLeaf0x80000008)?;

        // CPUID Fn8000_0008_ECX[15:12] (Field Name: ApicIdSize)
        // APIC ID size. The number of bits in the initial APIC20[ApicId] value that indicate
        // logical processor ID within a package. The size of this field determines the
        // maximum number of logical processors (MNLP) that the package could
        // theoretically support, and not the actual number of logical processors that are
        // implemented or enabled in the package, as indicated by CPUID
        // Fn8000_0008_ECX[NC]. A value of zero indicates that legacy methods must be
        // used to determine the maximum number of logical processors, as indicated by
        // CPUID Fn8000_0008_ECX[NC].
        set_range(&mut leaf_80000008.result.ecx, 12..=15, THREAD_ID_MAX_SIZE).unwrap();

        // CPUID Fn8000_0008_ECX[7:0] (Field Name: NC)
        // Number of physical threads - 1. The number of threads in the processor is NT+1
        // (e.g., if NT = 0, then there is one thread). See “Legacy Method” on page 633.
        let sub = cpu_count
            .checked_sub(1)
            .ok_or(FeatureEntryError::NumberOfPhysicalThreadsOverflow)?;
        set_range(&mut leaf_80000008.result.ecx, 0..=7, u32::from(sub))
            .map_err(FeatureEntryError::NumberOfPhysicalThreads)?;

        Ok(())
    }

    /// Update extended cache topology entry.
    #[allow(clippy::unwrap_in_result, clippy::unwrap_used)]
    fn update_extended_cache_topology_entry(
        &mut self,
        cpu_count: u8,
        cpus_per_core: u8,
    ) -> Result<(), ExtendedCacheTopologyError> {
        for i in 0.. {
            if let Some(subleaf) = self.get_mut(&CpuidKey::subleaf(0x8000001d, i)) {
                // CPUID Fn8000_001D_EAX_x[7:5] (Field Name: CacheLevel)
                // Cache level. Identifies the level of this cache. Note that the enumeration value
                // is not necessarily equal to the cache level.
                // ```text
                // Bits Description
                // 000b Reserved.
                // 001b Level 1
                // 010b Level 2
                // 011b Level 3
                // 111b-100b Reserved.
                // ```
                let cache_level = get_range(subleaf.result.eax, 5..=7);

                // CPUID Fn8000_001D_EAX_x[25:14] (Field Name: NumSharingCache)
                // Specifies the number of logical processors sharing the cache enumerated by N,
                // the value passed to the instruction in ECX. The number of logical processors
                // sharing this cache is the value of this field incremented by 1. To determine
                // which logical processors are sharing a cache, determine a Share
                // Id for each processor as follows:
                //
                // ShareId = LocalApicId >> log2(NumSharingCache+1)
                //
                // Logical processors with the same ShareId then share a cache. If
                // NumSharingCache+1 is not a power of two, round it up to the next power of two.

                match cache_level {
                    // L1 & L2 Cache
                    // The L1 & L2 cache is shared by at most 2 hyper-threads
                    1 | 2 => {
                        // SAFETY: We know `cpus_per_core > 0` therefore this is always safe.
                        let sub = u32::from(cpus_per_core.checked_sub(1).unwrap());
                        set_range(&mut subleaf.result.eax, 14..=25, sub)
                            .map_err(|err| ExtendedCacheTopologyError::NumSharingCache(i, err))?;
                    }
                    // L3 Cache
                    // The L3 cache is shared among all the logical threads
                    3 => {
                        let sub = cpu_count
                            .checked_sub(1)
                            .ok_or(ExtendedCacheTopologyError::NumSharingCacheOverflow(i))?;
                        set_range(&mut subleaf.result.eax, 14..=25, u32::from(sub))
                            .map_err(|err| ExtendedCacheTopologyError::NumSharingCache(i, err))?;
                    }
                    _ => (),
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Update extended apic id entry
    #[allow(clippy::unwrap_used, clippy::unwrap_in_result)]
    fn update_extended_apic_id_entry(
        &mut self,
        cpu_index: u8,
        cpus_per_core: u8,
    ) -> Result<(), ExtendedApicIdError> {
        /// 1 node per processor.
        const NODES_PER_PROCESSOR: u32 = 0;

        // When hyper-threading is enabled each pair of 2 consecutive logical CPUs
        // will have the same core id since they represent 2 threads in the same core.
        // For Example:
        // logical CPU 0 -> core id: 0
        // logical CPU 1 -> core id: 0
        // logical CPU 2 -> core id: 1
        // logical CPU 3 -> core id: 1
        //
        // SAFETY: We know `cpus_per_core != 0` therefore this is always safe.
        let core_id = u32::from(cpu_index.checked_div(cpus_per_core).unwrap());

        let leaf_8000001e = self
            .get_mut(&CpuidKey::leaf(0x8000001e))
            .ok_or(ExtendedApicIdError::MissingLeaf0x8000001e)?;

        // CPUID Fn8000_001E_EAX[31:0] (Field Name: ExtendedApicId)
        // Extended APIC ID. If MSR0000_001B[ApicEn] = 0, this field is reserved.
        set_range(&mut leaf_8000001e.result.eax, 0..=31, u32::from(cpu_index))
            .map_err(ExtendedApicIdError::ExtendedApicId)?;

        // CPUID Fn8000_001E_EBX[7:0] (Field Name: ComputeUnitId)
        // Compute unit ID. Identifies a Compute Unit, which may be one or more physical cores that
        // each implement one or more logical processors.
        set_range(&mut leaf_8000001e.result.ebx, 0..=7, core_id)
            .map_err(ExtendedApicIdError::ComputeUnitId)?;

        // CPUID Fn8000_001E_EBX[15:8] (Field Name: ThreadsPerComputeUnit)
        // Threads per compute unit (zero-based count). The actual number of threads
        // per compute unit is the value of this field + 1. To determine which logical
        // processors (threads) belong to a given Compute Unit, determine a ShareId
        // for each processor as follows:
        //
        // ShareId = LocalApicId >> log2(ThreadsPerComputeUnit+1)
        //
        // Logical processors with the same ShareId then belong to the same Compute
        // Unit. (If ThreadsPerComputeUnit+1 is not a power of two, round it up to the
        // next power of two).
        //
        // SAFETY: We know `cpus_per_core > 0` therefore this is always safe.
        let sub = u32::from(cpus_per_core.checked_sub(1).unwrap());
        set_range(&mut leaf_8000001e.result.ebx, 8..=15, sub)
            .map_err(ExtendedApicIdError::ThreadPerComputeUnit)?;

        // CPUID Fn8000_001E_ECX[10:8] (Field Name: NodesPerProcessor)
        // Specifies the number of nodes in the package/socket in which this logical
        // processor resides. Node in this context corresponds to a processor die.
        // Encoding is N-1, where N is the number of nodes present in the socket.
        //
        // SAFETY: We know the value always fits within the range and thus is always safe.
        // Set nodes per processor.
        set_range(&mut leaf_8000001e.result.ecx, 8..=10, NODES_PER_PROCESSOR).unwrap();

        // CPUID Fn8000_001E_ECX[7:0] (Field Name: NodeId)
        // Specifies the ID of the node containing the current logical processor. NodeId
        // values are unique across the system.
        //
        // Put all the cpus in the same node.
        set_range(&mut leaf_8000001e.result.ecx, 0..=7, 0).unwrap();

        Ok(())
    }

    /// Update brand string entry
    fn update_brand_string_entry(&mut self) -> Result<(), NormalizeCpuidError> {
        self.apply_brand_string(Self::DEFAULT_BRAND_STRING)
            .map_err(NormalizeCpuidError::BrandString)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use std::collections::BTreeMap;

    use super::*;
    use crate::cpu_config::x86_64::cpuid::AmdCpuid;

    #[test]
    fn test_update_structured_extended_entry_invalid() {
        // `update_structured_extended_entry()` should exit with MissingLeaf0x7Subleaf0 error for
        // CPUID lacking leaf 0x7 / subleaf 0.
        let mut cpuid = AmdCpuid(BTreeMap::new());
        assert_eq!(
            cpuid.update_structured_extended_entry().unwrap_err(),
            NormalizeCpuidError::MissingLeaf0x7Subleaf0
        );
    }

    #[test]
    fn test_update_structured_extended_entry_valid() {
        // `update_structured_extended_entry()` should succeed for CPUID having leaf 0x7 / subleaf
        // 0, and bit 29 of EDX (IA32_ARCH_CAPABILITIES MSR enumeration) should be disabled.
        let mut cpuid = AmdCpuid(BTreeMap::from([(
            CpuidKey {
                leaf: 0x7,
                subleaf: 0x0,
            },
            CpuidEntry {
                flags: KvmCpuidFlags::SIGNIFICANT_INDEX,
                result: CpuidRegisters {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: u32::MAX,
                },
            },
        )]));
        cpuid.update_structured_extended_entry().unwrap();
        assert_eq!(
            cpuid
                .get(&CpuidKey {
                    leaf: 0x7,
                    subleaf: 0x0
                })
                .unwrap()
                .result
                .edx
                & (1 << 29),
            0
        );
    }
}
