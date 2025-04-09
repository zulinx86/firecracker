# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CPU topology emulation feature."""

# pylint: disable=too-many-lines

import csv
import io
import os
import platform
import re
import shutil
import sys
import time
from difflib import unified_diff
from pathlib import Path

import pytest

import framework.utils_cpuid as cpuid_utils
from framework import utils
from framework.defs import SUPPORTED_HOST_KERNELS
from framework.properties import global_props
from framework.utils_cpu_templates import get_cpu_template_name

PLATFORM = platform.machine()
UNSUPPORTED_HOST_KERNEL = (
    utils.get_kernel_version(level=1) not in SUPPORTED_HOST_KERNELS
)
DATA_FILES = Path("./data/msr")

pytestmark = pytest.mark.skipif(
    global_props.cpu_architecture != "x86_64", reason="Only run in x86_64"
)


def read_msr_csv(fd):
    """Read a CSV of MSRs"""
    csvin = csv.DictReader(fd)
    return list(csvin)


def clean_and_mkdir(dir_path):
    """
    Create a clean directory
    """
    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)


def _check_cpuid_x86(test_microvm, expected_cpu_count, expected_htt):
    expected_cpu_features = {
        "maximum IDs for CPUs in pkg": f"{expected_cpu_count:#x} ({expected_cpu_count})",
        "CLFLUSH line size": "0x8 (8)",
        "hypervisor guest status": "true",
        "hyper-threading / multi-core supported": expected_htt,
    }

    cpuid_utils.check_guest_cpuid_output(
        test_microvm, "cpuid -1", None, "=", expected_cpu_features
    )


def _check_extended_cache_features(vm):
    l3_params = cpuid_utils.get_guest_cpuid(vm, "0x80000006")[(0x80000006, 0, "edx")]

    # fmt: off
    line_size     = (l3_params >>  0) & 0xFF
    lines_per_tag = (l3_params >>  8) & 0xF
    assoc         = (l3_params >> 12) & 0xF
    cache_size    = (l3_params >> 18) & 0x3FFF
    # fmt: on

    assert line_size > 0
    assert lines_per_tag == 0x1  # This is hardcoded in the AMD spec
    assert assoc == 0x9  # This is hardcoded in the AMD spec
    assert cache_size > 0


def skip_test_based_on_artifacts(snapshot_artifacts_dir):
    """
    It is possible that some X template is not supported on
    the instance where the snapshots were created and,
    snapshot is loaded on an instance where X is supported. This
    results in error since restore doesn't find the file to load.
    e.g. let's suppose snapshot is created on Skylake and restored
    on Cascade Lake. So, the created artifacts could just be:
    snapshot_artifacts/wrmsr/vmlinux-5.10/T2S
    but the restore test would fail because the files in
    snapshot_artifacts/wrmsr/vmlinux-5.10/T2CL won't be available.
    To avoid this we make an assumption that if template directory
    does not exist then snapshot was not created for that template
    and we skip the test.
    """
    if not Path.exists(snapshot_artifacts_dir):
        reason = f"\n Since {snapshot_artifacts_dir} does not exist \
                we skip the test assuming that snapshot was not"
        pytest.skip(re.sub(" +", " ", reason))


@pytest.mark.parametrize(
    "num_vcpus",
    [1, 2, 16],
)
@pytest.mark.parametrize(
    "htt",
    [True, False],
)
def test_cpuid(uvm_plain_any, num_vcpus, htt):
    """
    Check the CPUID for a microvm with the specified config.
    """
    vm = uvm_plain_any
    vm.spawn()
    vm.basic_config(vcpu_count=num_vcpus, smt=htt)
    vm.add_net_iface()
    vm.start()
    _check_cpuid_x86(vm, num_vcpus, "true" if num_vcpus > 1 else "false")


@pytest.mark.skipif(
    cpuid_utils.get_cpu_vendor() != cpuid_utils.CpuVendor.AMD,
    reason="L3 cache info is only present in 0x80000006 for AMD",
)
def test_extended_cache_features(uvm_plain_any):
    """
    Check extended cache features (leaf 0x80000006).
    """
    vm = uvm_plain_any
    vm.spawn()
    vm.basic_config()
    vm.add_net_iface()
    vm.start()
    _check_extended_cache_features(vm)


def test_brand_string(uvm_plain_any):
    """
    Ensure good formatting for the guest brand string.

    * For Intel CPUs, the guest brand string should be:
        Intel(R) Xeon(R) Processor @ {host frequency}
    or
        Intel(R) Xeon(R) Processor
    where {host frequency} is the frequency reported by the host CPUID
    (e.g. 4.01GHz)
    * For AMD CPUs, the guest brand string should be:
        AMD EPYC
    * For other CPUs, the guest brand string should be:
        ""
    """
    test_microvm = uvm_plain_any
    test_microvm.spawn()

    test_microvm.basic_config(vcpu_count=1)
    test_microvm.add_net_iface()
    test_microvm.start()

    guest_cmd = "cat /proc/cpuinfo | grep 'model name' | head -1"
    _, stdout, stderr = test_microvm.ssh.run(guest_cmd)
    assert stderr == ""

    line = stdout.rstrip()
    mo = re.search("^model name\\s+:\\s+(.+)$", line)
    assert mo
    guest_brand_string = mo.group(1)
    assert guest_brand_string

    cpu_vendor = cpuid_utils.get_cpu_vendor()
    if cpu_vendor == cpuid_utils.CpuVendor.AMD:
        # Assert the model name matches "AMD EPYC"
        mo = re.search("model name.*: AMD EPYC", stdout)
        assert mo
    elif cpu_vendor == cpuid_utils.CpuVendor.INTEL:
        # Get host frequency
        cif = open("/proc/cpuinfo", "r", encoding="utf-8")
        cpu_info = cif.read()
        mo = re.search("model name.*:.* ([0-9]*.[0-9]*[G|M|T]Hz)", cpu_info)
        # Skip if host frequency is not reported
        if mo is None:
            return
        host_frequency = mo.group(1)

        # Assert the model name matches "Intel(R) Xeon(R) Processor @ "
        mo = re.search(
            "model name.*: Intel\\(R\\) Xeon\\(R\\) Processor @ ([0-9]*.[0-9]*[T|G|M]Hz)",
            stdout,
        )
        assert mo
        # Get the frequency
        guest_frequency = mo.group(1)

        # Assert the guest frequency matches the host frequency
        assert host_frequency == guest_frequency
    else:
        assert False


# From the `Intel® 64 Architecture x2APIC Specification`
# (https://courses.cs.washington.edu/courses/cse451/24wi/documentation/x2apic.pdf):
# > The X2APIC MSRs cannot to be loaded and stored on VMX transitions. A VMX transition fails
# > if the VMM has specified that the transition should access any MSRs in the address range
# > from 0000_0800H to 0000_08FFH
X2APIC_MSRS = [hex(i) for i in range(0x0000_0800, 0x0000_08FF + 1)]


# Some MSR values should not be checked since they can change at guest runtime
# and between different boots.
# Current exceptions:
# * FS and GS change on task switch and arch_prctl.
# * TSC is different for each guest.
# * MSR_{C, L}STAR used for SYSCALL/SYSRET; can be different between guests.
# * MSR_IA32_SYSENTER_E{SP, IP} used for SYSENTER/SYSEXIT; same as above.
# * MSR_KVM_{WALL, SYSTEM}_CLOCK addresses for struct pvclock_* can be different.
# * MSR_IA32_TSX_CTRL is not available to read/write via KVM (known limitation).
#
# More detailed information about MSRs can be found in the Intel® 64 and IA-32
# Architectures Software Developer’s Manual - Volume 4: Model-Specific Registers
# Check `arch_gen/src/x86/msr_idex.rs` and `msr-index.h` in upstream Linux
# for symbolic definitions.
# fmt: off
MSR_EXCEPTION_LIST = [
    "0x10",        # MSR_IA32_TSC
    "0x11",        # MSR_KVM_WALL_CLOCK
    "0x12",        # MSR_KVM_SYSTEM_TIME
    "0x122",       # MSR_IA32_TSX_CTRL
    "0x175",       # MSR_IA32_SYSENTER_ESP
    "0x176",       # MSR_IA32_SYSENTER_EIP
    "0x6e0",       # MSR_IA32_TSC_DEADLINE
    "0xc0000082",  # MSR_LSTAR
    "0xc0000083",  # MSR_CSTAR
    "0xc0000100",  # MSR_FS_BASE
    "0xc0000101",  # MSR_GS_BASE
    # MSRs below are required only on T2A, however,
    # we are adding them to the common exception list to keep things simple
    "0x834"     ,  # LVT Performance Monitor Interrupt Register
    "0xc0010007",  # MSR_K7_PERFCTR3
    "0xc001020b",  # Performance Event Counter MSR_F15H_PERF_CTR5
    "0xc0011029",  # MSR_F10H_DECFG also referred to as MSR_AMD64_DE_CFG
    "0x830"     ,  # IA32_X2APIC_ICR is interrupt command register and,
                   # bit 0-7 represent interrupt vector that varies.
    "0x83f"     ,  # IA32_X2APIC_SELF_IPI
                   # A self IPI is semantically identical to an
                   # inter-processor interrupt sent via the ICR,
                   # with a Destination Shorthand of Self,
                   # Trigger Mode equal to Edge,
                   # and a Delivery Mode equal to Fixed.
                   # bit 0-7 represent interrupt vector that varies.
] + X2APIC_MSRS
# fmt: on


MSR_SUPPORTED_TEMPLATES = ["T2A", "T2CL", "T2S", "SPR_TO_T2_5.10", "SPR_TO_T2_6.1"]


@pytest.mark.timeout(900)
@pytest.mark.nonci
def test_cpu_rdmsr(
    microvm_factory, cpu_template_any, guest_kernel, rootfs, results_dir
):
    """
    Test MSRs that are available to the guest.

    This test boots a uVM and tries to read a set of MSRs from the guest.
    The guest MSR list is compared against a list of MSRs that are expected
    when running on a particular combination of host CPU model, host kernel,
    guest kernel and CPU template.

    The list is dependent on:
    * host CPU model, since some MSRs are passed through from the host in some
      CPU templates
    * host kernel version, since firecracker relies on MSR emulation provided
      by KVM
    * guest kernel version, since some MSRs are writable from guest uVMs and
      different guest kernels might set different values
    * CPU template, since enabled CPUIDs are different between CPU templates
      and some MSRs are not available if CPUID features are disabled

    This comparison helps validate that defaults have not changed due to
    emulation implementation changes by host kernel patches and CPU templates.

    TODO: This validates T2S, T2CL and T2A templates. Since T2 and C3 did not
    set the ARCH_CAPABILITIES MSR, the value of that MSR is different between
    different host CPU types (see Github PR #3066). So we can either:
    * add an exceptions for different template types when checking values
    * deprecate T2 and C3 since they are somewhat broken

    Testing matrix:
    - All supported guest kernels and rootfs
    - Microvm: 1vCPU with 1024 MB RAM
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in MSR_SUPPORTED_TEMPLATES:
        pytest.skip(f"This test does not support {cpu_template_name} template.")

    vcpus, guest_mem_mib = 1, 1024
    vm = microvm_factory.build(guest_kernel, rootfs, monitor_memory=False)
    vm.spawn()
    vm.add_net_iface()
    vm.basic_config(vcpu_count=vcpus, mem_size_mib=guest_mem_mib)
    vm.set_cpu_template(cpu_template_any)
    vm.start()
    vm.ssh.scp_put(DATA_FILES / "msr_reader.sh", "/tmp/msr_reader.sh")
    _, stdout, stderr = vm.ssh.run("/tmp/msr_reader.sh", timeout=None)
    assert stderr == ""

    # Load results read from the microvm
    guest_recs = read_msr_csv(io.StringIO(stdout))

    # Load baseline
    host_cpu = global_props.cpu_codename
    host_kv = global_props.host_linux_version
    guest_kv = re.search(r"vmlinux-(\d+\.\d+)", guest_kernel.name).group(1)
    baseline_file_name = (
        f"msr_list_{cpu_template_name}_{host_cpu}_{host_kv}host_{guest_kv}guest.csv"
    )
    # save it as an artifact, so we don't have to manually launch an instance to
    # get a baseline
    save_msrs = results_dir / baseline_file_name
    save_msrs.write_text(stdout)

    # Load baseline
    baseline_file_path = DATA_FILES / baseline_file_name
    baseline_recs = read_msr_csv(baseline_file_path.open())

    check_msrs_are_equal(baseline_recs, guest_recs)


# These names need to be consistent across the two parts of the snapshot-restore test
# that spans two instances (one that takes a snapshot and one that restores from it)
# fmt: off
SNAPSHOT_RESTORE_SHARED_NAMES = {
    "snapshot_artifacts_root_dir_wrmsr": "snapshot_artifacts/wrmsr",
    "snapshot_artifacts_root_dir_cpuid": "snapshot_artifacts/cpuid",
    "msr_reader_host_fname":             DATA_FILES / "msr_reader.sh",
    "msr_reader_guest_fname":            "/tmp/msr_reader.sh",
    "msrs_before_fname":                 "msrs_before.txt",
    "msrs_after_fname":                  "msrs_after.txt",
    "cpuid_before_fname":                "cpuid_before.txt",
    "cpuid_after_fname":                 "cpuid_after.txt",
}
# fmt: on


def dump_msr_state_to_file(dump_fname, ssh_conn, shared_names):
    """
    Read MSR state via SSH and dump it into a file.
    """
    ssh_conn.scp_put(
        shared_names["msr_reader_host_fname"], shared_names["msr_reader_guest_fname"]
    )
    _, stdout, stderr = ssh_conn.run(
        shared_names["msr_reader_guest_fname"], timeout=None
    )
    assert stderr == ""

    with open(dump_fname, "w", encoding="UTF-8") as file:
        file.write(stdout)


@pytest.mark.skipif(
    UNSUPPORTED_HOST_KERNEL,
    reason=f"Supported kernels are {SUPPORTED_HOST_KERNELS}",
)
@pytest.mark.timeout(900)
@pytest.mark.nonci
def test_cpu_wrmsr_snapshot(microvm_factory, guest_kernel, rootfs, cpu_template_any):
    """
    This is the first part of the test verifying
    that MSRs retain their values after restoring from a snapshot.

    This function makes MSR value modifications according to the
    ./data/msr/wrmsr_list.txt file.

    Before taking a snapshot, MSR values are dumped into a text file.
    After restoring from the snapshot on another instance, the MSRs are
    dumped again and their values are compared to previous.
    Some MSRs are not inherently supposed to retain their values, so they
    form an MSR exception list.

    This part of the test is responsible for taking a snapshot and publishing
    its files along with the `before` MSR dump.
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in MSR_SUPPORTED_TEMPLATES:
        pytest.skip(f"This test does not support {cpu_template_name} template.")

    shared_names = SNAPSHOT_RESTORE_SHARED_NAMES

    vcpus, guest_mem_mib = 1, 1024
    vm = microvm_factory.build(guest_kernel, rootfs, monitor_memory=False)
    vm.spawn()
    vm.add_net_iface()
    vm.basic_config(
        vcpu_count=vcpus,
        mem_size_mib=guest_mem_mib,
        track_dirty_pages=True,
        boot_args="msr.allow_writes=on",
    )
    vm.set_cpu_template(cpu_template_any)
    vm.start()

    # Make MSR modifications
    msr_writer_host_fname = DATA_FILES / "msr_writer.sh"
    msr_writer_guest_fname = "/tmp/msr_writer.sh"
    vm.ssh.scp_put(msr_writer_host_fname, msr_writer_guest_fname)

    wrmsr_input_host_fname = DATA_FILES / "wrmsr_list.txt"
    wrmsr_input_guest_fname = "/tmp/wrmsr_input.txt"
    vm.ssh.scp_put(wrmsr_input_host_fname, wrmsr_input_guest_fname)

    _, _, stderr = vm.ssh.run(
        f"{msr_writer_guest_fname} {wrmsr_input_guest_fname}", timeout=None
    )
    assert stderr == ""

    # Dump MSR state to a file that will be published to S3 for the 2nd part of the test
    snapshot_artifacts_dir = (
        Path(shared_names["snapshot_artifacts_root_dir_wrmsr"])
        / guest_kernel.name
        / get_cpu_template_name(cpu_template_any, with_type=True)
    )
    clean_and_mkdir(snapshot_artifacts_dir)

    msrs_before_fname = snapshot_artifacts_dir / shared_names["msrs_before_fname"]

    dump_msr_state_to_file(msrs_before_fname, vm.ssh, shared_names)
    # On T2A, the restore test fails with error "cannot allocate memory" so,
    # adding delay below as a workaround to unblock the tests for now.
    # TODO: Debug the issue and remove this delay. Create below issue to track this:
    # https://github.com/firecracker-microvm/firecracker/issues/3453
    time.sleep(0.25)

    # Take a snapshot
    snapshot = vm.snapshot_diff()
    # Copy snapshot files to be published to S3 for the 2nd part of the test
    snapshot.save_to(snapshot_artifacts_dir)


def check_msrs_are_equal(before_recs, after_recs):
    """
    Checks that reported MSRs and their values in the files are equal.
    """

    before = {x["MSR_ADDR"]: x["VALUE"] for x in before_recs}
    after = {x["MSR_ADDR"]: x["VALUE"] for x in after_recs}
    # We first want to see if the same set of MSRs are exposed in the microvm.
    all_msrs = set(before.keys()) | set(after.keys())

    changes = 0
    for msr in all_msrs:
        if msr in before and msr not in after:
            print(f"MSR removed {msr} before={before[msr]}")
            changes += 1
        elif msr not in before and msr in after:
            print(f"MSR added {msr} after={after[msr]}")
            changes += 1
        elif msr in MSR_EXCEPTION_LIST:
            continue
        elif before[msr] != after[msr]:
            # Compare values
            print(f"MSR changed {msr} before={before[msr]} after={after[msr]}")
            changes += 1
    assert changes == 0


@pytest.mark.skipif(
    UNSUPPORTED_HOST_KERNEL,
    reason=f"Supported kernels are {SUPPORTED_HOST_KERNELS}",
)
@pytest.mark.timeout(900)
@pytest.mark.nonci
def test_cpu_wrmsr_restore(microvm_factory, cpu_template_any, guest_kernel):
    """
    This is the second part of the test verifying
    that MSRs retain their values after restoring from a snapshot.

    Before taking a snapshot, MSR values are dumped into a text file.
    After restoring from the snapshot on another instance, the MSRs are
    dumped again and their values are compared to previous.
    Some MSRs are not inherently supposed to retain their values, so they
    form an MSR exception list.

    This part of the test is responsible for restoring from a snapshot and
    comparing two sets of MSR values.
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in MSR_SUPPORTED_TEMPLATES:
        pytest.skip(f"This test does not support {cpu_template_name} template.")

    shared_names = SNAPSHOT_RESTORE_SHARED_NAMES
    snapshot_artifacts_dir = (
        Path(shared_names["snapshot_artifacts_root_dir_wrmsr"])
        / guest_kernel.name
        / get_cpu_template_name(cpu_template_any, with_type=True)
    )

    skip_test_based_on_artifacts(snapshot_artifacts_dir)

    vm = microvm_factory.build()
    vm.spawn()
    vm.restore_from_path(snapshot_artifacts_dir, resume=True)

    # Dump MSR state to a file for further comparison
    msrs_after_fname = snapshot_artifacts_dir / shared_names["msrs_after_fname"]
    dump_msr_state_to_file(msrs_after_fname, vm.ssh, shared_names)
    msrs_before_fname = snapshot_artifacts_dir / shared_names["msrs_before_fname"]

    # Compare the two lists of MSR values and assert they are equal
    before_recs = read_msr_csv(msrs_before_fname.open())
    after_recs = read_msr_csv(msrs_after_fname.open())
    check_msrs_are_equal(before_recs, after_recs)


def dump_cpuid_to_file(dump_fname, ssh_conn):
    """
    Read CPUID via SSH and dump it into a file.
    """
    _, stdout, stderr = ssh_conn.run("cpuid --one-cpu")
    assert stderr == ""
    dump_fname.write_text(stdout, encoding="UTF-8")


@pytest.mark.skipif(
    UNSUPPORTED_HOST_KERNEL,
    reason=f"Supported kernels are {SUPPORTED_HOST_KERNELS}",
)
@pytest.mark.timeout(900)
@pytest.mark.nonci
def test_cpu_cpuid_snapshot(microvm_factory, guest_kernel, rootfs, cpu_template_any):
    """
    This is the first part of the test verifying
    that CPUID remains the same after restoring from a snapshot.

    Before taking a snapshot, CPUID is dumped into a text file.
    After restoring from the snapshot on another instance, the CPUID is
    dumped again and its content is compared to previous.

    This part of the test is responsible for taking a snapshot and publishing
    its files along with the `before` CPUID dump.
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in MSR_SUPPORTED_TEMPLATES:
        pytest.skip("This test does not support {cpu_template_name} template.")

    shared_names = SNAPSHOT_RESTORE_SHARED_NAMES

    vm = microvm_factory.build(
        kernel=guest_kernel,
        rootfs=rootfs,
    )
    vm.spawn()
    vm.add_net_iface()
    vm.basic_config(
        vcpu_count=1,
        mem_size_mib=1024,
        track_dirty_pages=True,
    )
    vm.set_cpu_template(cpu_template_any)
    vm.start()

    # Dump CPUID to a file that will be published to S3 for the 2nd part of the test
    snapshot_artifacts_dir = (
        Path(shared_names["snapshot_artifacts_root_dir_cpuid"])
        / guest_kernel.name
        / get_cpu_template_name(cpu_template_any, with_type=True)
    )
    clean_and_mkdir(snapshot_artifacts_dir)

    cpuid_before_fname = snapshot_artifacts_dir / shared_names["cpuid_before_fname"]

    dump_cpuid_to_file(cpuid_before_fname, vm.ssh)

    # Take a snapshot
    snapshot = vm.snapshot_diff()
    # Copy snapshot files to be published to S3 for the 2nd part of the test
    snapshot.save_to(snapshot_artifacts_dir)


def check_cpuid_is_equal(before_cpuid_fname, after_cpuid_fname):
    """
    Checks that CPUID dumps in the files are equal.
    """
    with open(before_cpuid_fname, "r", encoding="UTF-8") as file:
        before = file.readlines()
    with open(after_cpuid_fname, "r", encoding="UTF-8") as file:
        after = file.readlines()

    diff = sys.stdout.writelines(unified_diff(before, after))

    assert not diff, f"\n\n{diff}"


@pytest.mark.skipif(
    UNSUPPORTED_HOST_KERNEL,
    reason=f"Supported kernels are {SUPPORTED_HOST_KERNELS}",
)
@pytest.mark.timeout(900)
@pytest.mark.nonci
def test_cpu_cpuid_restore(microvm_factory, guest_kernel, cpu_template_any):
    """
    This is the second part of the test verifying
    that CPUID remains the same after restoring from a snapshot.

    Before taking a snapshot, CPUID is dumped into a text file.
    After restoring from the snapshot on another instance, the CPUID is
    dumped again and compared to previous.

    This part of the test is responsible for restoring from a snapshot and
    comparing two CPUIDs.
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in MSR_SUPPORTED_TEMPLATES:
        pytest.skip("This test does not support {cpu_template_name} template.")

    shared_names = SNAPSHOT_RESTORE_SHARED_NAMES
    snapshot_artifacts_dir = (
        Path(shared_names["snapshot_artifacts_root_dir_cpuid"])
        / guest_kernel.name
        / get_cpu_template_name(cpu_template_any, with_type=True)
    )

    skip_test_based_on_artifacts(snapshot_artifacts_dir)

    vm = microvm_factory.build()
    vm.spawn()
    vm.restore_from_path(snapshot_artifacts_dir, resume=True)

    # Dump CPUID to a file for further comparison
    cpuid_after_fname = snapshot_artifacts_dir / shared_names["cpuid_after_fname"]
    dump_cpuid_to_file(cpuid_after_fname, vm.ssh)

    # Compare the two lists of MSR values and assert they are equal
    check_cpuid_is_equal(
        snapshot_artifacts_dir / shared_names["cpuid_before_fname"],
        snapshot_artifacts_dir / shared_names["cpuid_after_fname"],
    )


def test_cpu_template(uvm_plain_any, cpu_template_any, microvm_factory):
    """
    Test masked and enabled cpu features against the expected template.

    This test checks that all expected masked features are not present in the
    guest and that expected enabled features are present for each of the
    supported CPU templates.
    """
    cpu_template_name = get_cpu_template_name(cpu_template_any)
    if cpu_template_name not in ["T2", "T2S", "SPR_TO_T2_5.10", "SPR_TO_T2_6.1" "C3"]:
        pytest.skip("This test does not support {cpu_template_name} template.")

    test_microvm = uvm_plain_any
    test_microvm.spawn()
    # Set template as specified in the `cpu_template` parameter.
    test_microvm.basic_config(
        vcpu_count=1,
        mem_size_mib=256,
    )
    test_microvm.set_cpu_template(cpu_template_any)
    test_microvm.add_net_iface()

    if cpuid_utils.get_cpu_vendor() != cpuid_utils.CpuVendor.INTEL:
        # We shouldn't be able to apply Intel templates on AMD hosts
        with pytest.raises(RuntimeError):
            test_microvm.start()
        return

    test_microvm.start()

    check_masked_features(test_microvm, cpu_template_name)
    check_enabled_features(test_microvm, cpu_template_name)

    # Check that cpu features are still correct
    # after snap/restore cycle.
    snapshot = test_microvm.snapshot_full()
    restored_vm = microvm_factory.build()
    restored_vm.spawn()
    restored_vm.restore_from_snapshot(snapshot, resume=True)
    check_masked_features(restored_vm, cpu_template_name)
    check_enabled_features(restored_vm, cpu_template_name)


def check_masked_features(test_microvm, cpu_template):
    """Verify the masked features of the given template."""
    # fmt: off
    must_be_unset = []
    if cpu_template == "C3":
        must_be_unset = [
            (0x1, 0x0, "ecx",
                (1 << 2) |  # DTES64
                (1 << 3) |  # MONITOR
                (1 << 4) |  # DS_CPL_SHIFT
                (1 << 5) |  # VMX
                (1 << 8) |  # TM2
                (1 << 10) | # CNXT_ID
                (1 << 11) | # SDBG
                (1 << 12) | # FMA
                (1 << 14) | # XTPR_UPDATE
                (1 << 15) | # PDCM
                (1 << 22)   # MOVBE
            ),
            (0x1, 0x0, "edx",
                (1 << 18) | # PSN
                (1 << 21) | # DS
                (1 << 22) | # ACPI
                (1 << 27) | # SS
                (1 << 29) | # TM
                (1 << 31)   # PBE
            ),
            (0x7, 0x0, "ebx",
                (1 << 2) |  # SGX
                (1 << 3) |  # BMI1
                (1 << 4) |  # HLE
                (1 << 5) |  # AVX2
                (1 << 8) |  # BMI2
                (1 << 10) | # INVPCID
                (1 << 11) | # RTM
                (1 << 12) | # RDT_M
                (1 << 14) | # MPX
                (1 << 15) | # RDT_A
                (1 << 16) | # AVX512F
                (1 << 17) | # AVX512DQ
                (1 << 18) | # RDSEED
                (1 << 19) | # ADX
                (1 << 21) | # AVX512IFMA
                (1 << 23) | # CLFLUSHOPT
                (1 << 24) | # CLWB
                (1 << 25) | # PT
                (1 << 26) | # AVX512PF
                (1 << 27) | # AVX512ER
                (1 << 28) | # AVX512CD
                (1 << 29) | # SHA
                (1 << 30) | # AVX512BW
                (1 << 31)   # AVX512VL
            ),
            (0x7, 0x0, "ecx",
                (1 << 1) |  # AVX512_VBMI
                (1 << 2) |  # UMIP
                (1 << 3) |  # PKU
                (1 << 4) |  # OSPKE
                (1 << 11) | # AVX512_VNNI
                (1 << 14) | # AVX512_VPOPCNTDQ
                (1 << 16) | # LA57
                (1 << 22) | # RDPID
                (1 << 30)   # SGX_LC
            ),
            (0x7, 0x0, "edx",
                (1 << 2) |  # AVX512_4VNNIW
                (1 << 3)    # AVX512_4FMAPS
            ),
            (0xd, 0x0, "eax",
                (1 << 3) |  # MPX_STATE bit 0
                (1 << 4) |  # MPX_STATE bit 1
                (1 << 5) |  # AVX512_STATE bit 0
                (1 << 6) |  # AVX512_STATE bit 1
                (1 << 7) |  # AVX512_STATE bit 2
                (1 << 9)    # PKRU
            ),
            (0xd, 0x1, "eax",
                (1 << 1) |  # XSAVEC_SHIFT
                (1 << 2) |  # XGETBV_SHIFT
                (1 << 3)    # XSAVES_SHIFT
            ),
            (0x80000001, 0x0, "ecx",
                (1 << 5) |  # LZCNT
                (1 << 8)    # PREFETCH
            ),
            (0x80000001, 0x0, "edx",
                (1 << 26)   # PDPE1GB
            ),
        ]
    elif cpu_template in ("T2", "T2S"):
        must_be_unset = [
            (0x1, 0x0, "ecx",
                (1 << 2) |  # DTES64
                (1 << 3) |  # MONITOR
                (1 << 4) |  # DS_CPL_SHIFT
                (1 << 5) |  # VMX
                (1 << 6) |  # SMX
                (1 << 7) |  # EIST
                (1 << 8) |  # TM2
                (1 << 10) | # CNXT_ID
                (1 << 11) | # SDBG
                (1 << 14) | # XTPR_UPDATE
                (1 << 15) | # PDCM
                (1 << 18)   # DCA
            ),
            (0x1, 0x0, "edx",
                (1 << 18) | # PSN
                (1 << 21) | # DS
                (1 << 22) | # ACPI
                (1 << 27) | # SS
                (1 << 29) | # TM
                (1 << 30) | # IA64
                (1 << 31)   # PBE
            ),
            (0x7, 0x0, "ebx",
                (1 << 2) |  # SGX
                (1 << 4) |  # HLE
                (1 << 11) | # RTM
                (1 << 12) | # RDT_M
                (1 << 14) | # MPX
                (1 << 15) | # RDT_A
                (1 << 16) | # AVX512F
                (1 << 17) | # AVX512DQ
                (1 << 18) | # RDSEED
                (1 << 19) | # ADX
                (1 << 21) | # AVX512IFMA
                (1 << 22) | # PCOMMIT
                (1 << 23) | # CLFLUSHOPT
                (1 << 24) | # CLWB
                (1 << 25) | # PT
                (1 << 26) | # AVX512PF
                (1 << 27) | # AVX512ER
                (1 << 28) | # AVX512CD
                (1 << 29) | # SHA
                (1 << 30) | # AVX512BW
                (1 << 31)   # AVX512VL
            ),
            (0x7, 0x0, "ecx",
                (1 << 1) |  # AVX512_VBMI
                (1 << 2) |  # UMIP
                (1 << 3) |  # PKU
                (1 << 4) |  # OSPKE
                (1 << 6) |  # AVX512_VBMI2
                (1 << 8) |  # GFNI
                (1 << 9) |  # VAES
                (1 << 10) | # VPCLMULQDQ
                (1 << 11) | # AVX512_VNNI
                (1 << 12) | # AVX512_BITALG
                (1 << 14) | # AVX512_VPOPCNTDQ
                (1 << 16) | # LA57
                (1 << 22) | # RDPID
                (1 << 30)   # SGX_LC
            ),
            (0x7, 0x0, "edx",
                (1 << 2) |  # AVX512_4VNNIW
                (1 << 3) |  # AVX512_4FMAPS
                (1 << 4) |  # FSRM
                (1 << 8)    # AVX512_VP2INTERSECT
            ),
            (0xd, 0x0, "eax",
                (1 << 3) |  # MPX_STATE bit 0
                (1 << 4) |  # MPX_STATE bit 1
                (1 << 5) |  # AVX512_STATE bit 0
                (1 << 6) |  # AVX512_STATE bit 1
                (1 << 7) |  # AVX512_STATE bit 2
                (1 << 9)    # PKRU
            ),
            (0xd, 0x1, "eax",
                (1 << 1) |  # XSAVEC_SHIFT
                (1 << 2) |  # XGETBV_SHIFT
                (1 << 3)    # XSAVES_SHIFT
            ),
            (0x80000001, 0x0, "ecx",
                (1 << 8) |  # PREFETCH
                (1 << 29)   # MWAIT_EXTENDED
            ),
            (0x80000001, 0x0, "edx",
                (1 << 26)   # PDPE1GB
            ),
            (0x80000008, 0x0, "ebx",
                (1 << 9)    # WBNOINVD
            )
        ]
    elif cpu_template in ["SPR_TO_T2_5.10", "SPR_TO_T2_6.1"]:
        must_be_unset = [
            (0x1, 0x0, "ecx",
                (1 << 2) |  # DTES64
                (1 << 3) |  # MONITOR
                (1 << 4) |  # DS-CPL
                (1 << 5) |  # VMX
                (1 << 6) |  # SMX
                (1 << 7) |  # EIST
                (1 << 8) |  # TM2
                (1 << 10) | # CNXT-ID
                (1 << 11) | # SDBG
                (1 << 14) | # XTPR_UPDATE
                (1 << 15) | # PDCM
                (1 << 18)   # DCA
            ),
            (0x1, 0x0, "edx",
                (1 << 18) | # PSN
                (1 << 21) | # DS
                (1 << 22) | # ACPI
                (1 << 27) | # SS
                (1 << 29) | # TM
                (1 << 30) | # IA64
                (1 << 31)   # PBE
            ),
            (0x7, 0x0, "ebx",
                (1 << 2) |  # SGX
                (1 << 4) |  # HLE
                (1 << 11) | # RTM
                (1 << 12) | # RDT-M
                (1 << 14) | # MPX
                (1 << 15) | # RDT-A
                (1 << 16) | # AVX512F
                (1 << 17) | # AVX512DQ
                (1 << 18) | # RDSEED
                (1 << 19) | # ADX
                (1 << 21) | # AVX512IFMA
                (1 << 22) | # PCOMMIT
                (1 << 23) | # CLFLUSHOPT
                (1 << 24) | # CLWB
                (1 << 25) | # PT
                (1 << 26) | # AVX512PF
                (1 << 27) | # AVX512ER
                (1 << 28) | # AVX512CD
                (1 << 29) | # SHA
                (1 << 30) | # AVX512BW
                (1 << 31)   # AVX512VL
            ),
            (0x7, 0x0, "ecx",
                (1 << 1) |  # AVX512_VBMI
                (1 << 2) |  # UMIP
                (1 << 3) |  # PKU
                (1 << 4) |  # OSPKE
                (1 << 6) |  # AVX512_VBMI2
                (1 << 8) |  # GFNI
                (1 << 9) |  # VAES
                (1 << 10) | # VPCLMULQDQ
                (1 << 11) | # AVX512_VNNI
                (1 << 12) | # AVX512_BITALG
                (1 << 14) | # AVX512_VPOPCNTDQ
                (1 << 16) | # LA57
                (1 << 22) | # RDPID
                (1 << 24) | # BUS_LOCK_DETECT
                (1 << 25) | # CLDEMOTE
                (1 << 27) | # MOVDIRI
                (1 << 28) | # MOVDIR64B
                (1 << 30)   # SGX_LC
            ),
            (0x7, 0x0, "edx",
                (1 << 2) |  # AVX512_4VNNIW
                (1 << 3) |  # AVX512_4FMAPS
                (1 << 4) |  # FSRM
                (1 << 8) |  # AVX512_VP2INTERSECT
                (1 << 14) | # SERIALIZE
                (1 << 16) | # TSXLDTRK
                (1 << 22) | # AMX-BF16
                (1 << 23) | # AVX512_FP16
                (1 << 24) | # AMX-TILE
                (1 << 25)   # AMX-INT8
            ),
            (0x7, 0x1, "eax",
                (1 << 4) |  # AVX-VNI
                (1 << 5)    # AVX512_BF16
            ),
            # Note that we don't intentionally mask hardware security features.
            # - IPRED_CTRL: CPUID.(EAX=07H,ECX=2):EDX[1]
            # - RRSBA_CTRL: CPUID.(EAX=07H,ECX=2):EDX[2]
            # - BHI_CTRL: CPUID.(EAX=07H,ECX=2):EDX[4]
            (0xd, 0x0, "eax",
                (1 << 3) |  # MPX state bit 0
                (1 << 4) |  # MPX state bit 1
                (1 << 5) |  # AVX-512 state bit 0
                (1 << 6) |  # AVX-512 state bit 1
                (1 << 7) |  # AVX-512 state bit 2
                (1 << 9) |  # PKRU state
                (1 << 17) | # AMX TILECFG state
                (1 << 18)   # AMX TILEDATA state
            ),
            (0xd, 0x1, "eax",
                (1 << 1) |  # XSAVEC
                (1 << 2) |  # XGETBV with ECX=1
                (1 << 3) |  # XSAVES/XRSTORS and IA32_XSS
                (1 << 4)    # XFD
            ),
            (0xd, 0x11, "eax", (1 << 32) - 1), # AMX TILECFG XSTATE leaf, EAX
            (0xd, 0x11, "ebx", (1 << 32) - 1), # AMX TILECFG XSTATE leaf, EBX
            (0xd, 0x11, "ecx", (1 << 32) - 1), # AMX TILECFG XSTATE leaf, ECX
            (0xd, 0x11, "edx", (1 << 32) - 1), # AMX TILECFG XSTATE leaf, EDX
            (0xd, 0x12, "eax", (1 << 32) - 1), # AMX TILEDATA XSTATE leaf, EAX
            (0xd, 0x12, "ebx", (1 << 32) - 1), # AMX TILEDATA XSTATE leaf, EBX
            (0xd, 0x12, "ecx", (1 << 32) - 1), # AMX TILEDATA XSTATE leaf, ECX
            (0xd, 0x12, "edx", (1 << 32) - 1), # AMX TILEDATA XSTATE leaf, EDX
            (0x1d, 0x0, "eax", (1 << 32) - 1), # AMX Tile Information leaf, EAX
            (0x1d, 0x0, "ebx", (1 << 32) - 1), # AMX Tile Information leaf, EBX
            (0x1d, 0x0, "ecx", (1 << 32) - 1), # AMX Tile Information leaf, ECX
            (0x1d, 0x0, "edx", (1 << 32) - 1), # AMX Tile Information leaf, EDX
            (0x1d, 0x1, "eax", (1 << 32) - 1), # AMX Tile Palette 1 leaf, EAX
            (0x1d, 0x1, "ebx", (1 << 32) - 1), # AMX Tile Palette 1 leaf, EBX
            (0x1d, 0x1, "ecx", (1 << 32) - 1), # AMX Tile Palette 1 leaf, ECX
            (0x1d, 0x1, "edx", (1 << 32) - 1), # AMX TIle Palette 1 leaf, EDX
            (0x1e, 0x0, "eax", (1 << 32) - 1), # AMX TMUL Information leaf, EAX
            (0x1e, 0x0, "ebx", (1 << 32) - 1), # AMX TMUL Information leaf, EBX
            (0x1e, 0x0, "ecx", (1 << 32) - 1), # AMX TMUL Information leaf, ECX
            (0x1e, 0x0, "edx", (1 << 32) - 1), # AMX TMUL Information leaf, EDX
            (0x80000001, 0x0, "ecx",
                (1 << 8) |  # PREFETCHW
                (1 << 29)   # MWAITX / MONITORX
            ),
            (0x80000001, 0x0, "edx",
                (1 << 26)   # 1-GByte pages
            ),
            (0x80000008, 0x0, "ebx",
                (1 << 9)    # WBNOINVD
            )
        ]
    # fmt: on

    cpuid_utils.check_cpuid_feat_flags(
        test_microvm,
        [],
        must_be_unset,
    )


def check_enabled_features(test_microvm, cpu_template):
    """Test for checking that all expected features are enabled in guest."""
    enabled_list = {  # feature_info_1_edx
        "x87 FPU on chip": "true",
        "CMPXCHG8B inst.": "true",
        "VME: virtual-8086 mode enhancement": "true",
        "SSE extensions": "true",
        "SSE2 extensions": "true",
        "DE: debugging extensions": "true",
        "PSE: page size extensions": "true",
        "TSC: time stamp counter": "true",
        "RDMSR and WRMSR support": "true",
        "PAE: physical address extensions": "true",
        "MCE: machine check exception": "true",
        "APIC on chip": "true",
        "MMX Technology": "true",
        "SYSENTER and SYSEXIT": "true",
        "MTRR: memory type range registers": "true",
        "PTE global bit": "true",
        "FXSAVE/FXRSTOR": "true",
        "MCA: machine check architecture": "true",
        "CMOV: conditional move/compare instr": "true",
        "PAT: page attribute table": "true",
        "PSE-36: page size extension": "true",
        "CLFLUSH instruction": "true",
        # feature_info_1_ecx
        "PNI/SSE3: Prescott New Instructions": "true",
        "PCLMULDQ instruction": "true",
        "SSSE3 extensions": "true",
        "AES instruction": "true",
        "CMPXCHG16B instruction": "true",
        "PCID: process context identifiers": "true",
        "SSE4.1 extensions": "true",
        "SSE4.2 extensions": "true",
        "x2APIC: extended xAPIC support": "true",
        "POPCNT instruction": "true",
        "time stamp counter deadline": "true",
        "XSAVE/XSTOR states": "true",
        "OS-enabled XSAVE/XSTOR": "true",
        "AVX: advanced vector extensions": "true",
        "F16C half-precision convert instruction": "true",
        "RDRAND instruction": "true",
        "hypervisor guest status": "true",
        # thermal_and_power_mgmt
        "ARAT always running APIC timer": "true",
        # extended_features
        "FSGSBASE instructions": "true",
        "IA32_TSC_ADJUST MSR supported": "true",
        "SMEP supervisor mode exec protection": "true",
        "enhanced REP MOVSB/STOSB": "true",
        "SMAP: supervisor mode access prevention": "true",
        # xsave_0xd_0
        "x87 state": "true",
        "SSE state": "true",
        "AVX state": "true",
        # xsave_0xd_1
        "XSAVEOPT instruction": "true",
        # extended_080000001_edx
        "SYSCALL and SYSRET instructions": "true",
        "64-bit extensions technology available": "true",
        "execution disable": "true",
        "RDTSCP": "true",
        # intel_080000001_ecx
        "LAHF/SAHF supported in 64-bit mode": "true",
        # adv_pwr_mgmt
        "TscInvariant": "true",
    }

    cpuid_utils.check_guest_cpuid_output(
        test_microvm, "cpuid -1", None, "=", enabled_list
    )
    if cpu_template in ["T2", "SPR_TO_T2_5.10", "SPR_TO_T2_6.1"]:
        t2_enabled_features = {
            "FMA instruction": "true",
            "BMI1 instructions": "true",
            "BMI2 instructions": "true",
            "AVX2: advanced vector extensions 2": "true",
            "MOVBE instruction": "true",
            "INVPCID instruction": "true",
        }
        cpuid_utils.check_guest_cpuid_output(
            test_microvm, "cpuid -1", None, "=", t2_enabled_features
        )


def test_c3_on_skylake_show_warning(uvm_plain, cpu_template_any):
    """
    This test verifies that the warning message about MMIO stale data mitigation
    is displayed only on Intel Skylake with static C3 template.
    """
    uvm = uvm_plain
    uvm.spawn()
    uvm.basic_config(vcpu_count=2, mem_size_mib=256)
    uvm.add_net_iface()
    uvm.set_cpu_template(cpu_template_any)
    uvm.start()

    message = (
        "On processors that do not enumerate FBSDP_NO, PSDP_NO and "
        "SBDR_SSDP_NO on IA32_ARCH_CAPABILITIES MSR, the guest kernel "
        "does not apply the mitigation against MMIO stale data "
        "vulnerability."
    )

    if cpu_template_any == "C3" and global_props.cpu_codename == "INTEL_SKYLAKE":
        assert message in uvm.log_data
    else:
        assert message not in uvm.log_data


@pytest.mark.skipif(
    global_props.cpu_codename != "INTEL_SAPPHIRE_RAPIDS"
    or global_props.host_linux_version_tpl < (5, 17),
    reason="Intel AMX is only supported on Intel Sapphire Rapids and kernel v5.17+",
)
def test_intel_amx_reported_on_sapphire_rapids(
    microvm_factory, guest_kernel_linux_6_1, rootfs
):
    """
    Verifies that Intel AMX is reported on guest (v5.17+)
    """
    uvm = microvm_factory.build(guest_kernel_linux_6_1, rootfs)
    uvm.spawn()
    uvm.basic_config()
    uvm.add_net_iface()
    uvm.start()

    expected_dict = {
        "AMX-BF16: tile bfloat16 support": "true",  # CPUID.(EAX=07H,ECX=0):EDX[22]
        "AMX-TILE: tile architecture support": "true",  # CPUID.(EAX=07H,ECX=0):EDX[24]
        "AMX-INT8: tile 8-bit integer support": "true",  # CPUID.(EAX=07H,ECX=0):EDX[25]
        "AMX-FP16: FP16 tile operations": "false",  # CPUID.(EAX=07H,ECX=1):EAX[21], not supported on host as well
        "XTILECFG state": "true",  # CPUID.(EAX=0DH,ECX=0):EAX[17]
        "XTILEDATA state": "true",  # CPUID.(EAX=0DH,ECX=0):EAX[17]
    }
    cpuid_utils.check_guest_cpuid_output(
        uvm,
        "cpuid -1",
        None,
        "=",
        expected_dict,
    )
