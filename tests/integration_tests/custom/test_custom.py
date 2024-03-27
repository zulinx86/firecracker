# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom Tests"""

from framework import utils

def run_command_before_and_after_snapshot(
    microvm_factory,
    guest_kernel,
    rootfs,
    cmd,
):
    base_vm = microvm_factory.build(guest_kernel, rootfs)
    base_vm.spawn()
    base_vm.basic_config()
    base_vm.add_net_iface()
    base_vm.start()

    _ret, stdout, stderr = base_vm.ssh.run(cmd)
    print("before snapshot")
    print(stdout)
    print(stderr)

    # take a snapshot of the microVM and restore a microVM from the snapshot
    snapshot = base_vm.snapshot_full()
    restored_vm = microvm_factory.build()
    restored_vm.spawn()
    restored_vm.restore_from_snapshot(snapshot, resume=True)

    print("after snapshot")
    _ret, stdout, stderr = restored_vm.ssh.run(cmd)
    print(stdout)
    print(stderr)

# def test_cpuid_on_host():
#     """cpuid on host"""
#     # not working since devctr doesn't have cpuid package

def test_cpuid_on_guest(
    microvm_factory,
    guest_kernel,
    rootfs,
):
    """cpuid on guest"""
    run_command_before_and_after_snapshot(
        microvm_factory,
        guest_kernel,
        rootfs,
        "cpuid -r -1 | grep '0x00000007 0x00'"
    )

def test_ia32_arch_cap_on_host():
    """MSR on host"""
    _ret, stdout, stderr = utils.run_cmd("dd if=/dev/cpu/0/msr bs=8 count=1 skip=$((0x10a)) iflag=skip_bytes | od -t x8 -A n")
    print(stdout)
    print(stderr)

def test_ia32_arch_cap_on_guest(
    microvm_factory,
    guest_kernel,
    rootfs,
):
    """MSR on guest"""
    run_command_before_and_after_snapshot(
        microvm_factory,
        guest_kernel,
        rootfs,
        "dd if=/dev/cpu/0/msr bs=8 count=1 skip=$((0x10a)) iflag=skip_bytes | od -t x8 -A n"
    )

def test_ia32_tsx_ctrl_on_host():
    """MSR on host"""
    _ret, stdout, stderr = utils.run_cmd("dd if=/dev/cpu/0/msr bs=8 count=1 skip=$((0x122)) iflag=skip_bytes | od -t x8 -A n")
    print(stdout)
    print(stderr)

def test_ia32_tsx_ctrl_on_guest(
    microvm_factory,
    guest_kernel,
    rootfs,
):
    """MSR on guest"""
    run_command_before_and_after_snapshot(
        microvm_factory,
        guest_kernel,
        rootfs,
        "dd if=/dev/cpu/0/msr bs=8 count=1 skip=$((0x122)) iflag=skip_bytes | od -t x8 -A n"
    )

def test_taa_sysfs_on_host():
    """/sys/devices/system/cpu/vulnerabilities/tsx_async_abort on host"""
    _ret, stdout, stderr = utils.run_cmd("cat /sys/devices/system/cpu/vulnerabilities/tsx_async_abort")
    print(stdout)
    print(stderr)

def test_taa_sysfs_on_guest(
    microvm_factory,
    guest_kernel,
    rootfs,
):
    """/sys/devices/system/cpu/vulnerabilities/tsx_async_abort on guest"""
    run_command_before_and_after_snapshot(
        microvm_factory,
        guest_kernel,
        rootfs,
        "cat /sys/devices/system/cpu/vulnerabilities/tsx_async_abort"
    )

def test_rtm_sample_code_on_host():
    """attemp to run RTM sample code on host"""
    _ret, stdout, stderr = utils.run_cmd("./integration_tests/custom/a.out")
    print(stdout)
    print(stderr)

def test_rtm_sample_code_on_guest(
    microvm_factory,
    guest_kernel,
    rootfs_rw,
):
    """attempt to run RTM sample code on guest"""
    base_vm = microvm_factory.build(guest_kernel, rootfs_rw)
    base_vm.spawn()
    base_vm.basic_config()
    base_vm.add_net_iface()
    base_vm.start()

    local_path = "./integration_tests/custom/a.out"
    remote_path = "/a.out"
    base_vm.ssh.scp_put(local_path, remote_path)

    cmd = "chmod +x /a.out && /a.out"
    _ret, stdout, stderr = base_vm.ssh.run(cmd)
    print("before snapshot")
    print(stdout)
    print(stderr)

    # take a snapshot of the microVM and restore a microVM from the snapshot
    snapshot = base_vm.snapshot_full()
    restored_vm = microvm_factory.build()
    restored_vm.spawn()
    restored_vm.restore_from_snapshot(snapshot, resume=True)

    print("after snapshot")
    _ret, stdout, stderr = restored_vm.ssh.run(cmd)
    print(stdout)
    print(stderr)
