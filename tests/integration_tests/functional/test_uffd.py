# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test UFFD related functionality when resuming from snapshot."""

import os
import re

import pytest
import requests

from framework.utils import Timeout, check_output
from framework.utils_uffd import spawn_pf_handler, uffd_handler


@pytest.fixture(scope="function", name="snapshot")
def snapshot_fxt(microvm_factory, guest_kernel_linux_5_10, rootfs):
    """Create a snapshot of a microVM."""

    basevm = microvm_factory.build(guest_kernel_linux_5_10, rootfs)
    basevm.spawn()
    basevm.basic_config(vcpu_count=2, mem_size_mib=256)
    basevm.add_net_iface()

    basevm.start()

    # Create base snapshot.
    snapshot = basevm.snapshot_full()
    basevm.kill()

    yield snapshot


def test_bad_socket_path(uvm_plain, snapshot):
    """
    Test error scenario when socket path does not exist.
    """
    vm = uvm_plain
    vm.spawn()
    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)

    expected_msg = re.escape("Failed to connect to UDS Unix stream: No such file or directory (os error 2)")
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": "inexistent"},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


def test_unbinded_socket(uvm_plain, snapshot):
    """
    Test error scenario when PF handler has not yet called bind on socket.
    """
    vm = uvm_plain
    vm.spawn()

    jailed_vmstate = vm.create_jailed_resource(snapshot.vmstate)
    socket_path = os.path.join(vm.path, "firecracker-uffd.sock")
    check_output("touch {}".format(socket_path))
    jailed_sock_path = vm.create_jailed_resource(socket_path)

    expected_msg = re.escape("Failed to connect to UDS Unix stream: Connection refused (os error 111)")
    with pytest.raises(RuntimeError, match=expected_msg):
        vm.api.snapshot_load.put(
            mem_backend={"backend_type": "Uffd", "backend_path": jailed_sock_path},
            snapshot_path=jailed_vmstate,
        )

    vm.mark_killed()


def test_valid_handler(uvm_plain, snapshot):
    """
    Test valid uffd handler scenario.
    """
    vm = uvm_plain
    vm.memory_monitor = None
    vm.spawn()

    # Spawn page fault handler process.
    spawn_pf_handler(vm, uffd_handler("on_demand"), snapshot)

    vm.restore_from_snapshot(resume=True)


    # Verify if the restored guest works.
    vm.ssh.check_output("true")


def test_malicious_handler(uvm_plain, snapshot):
    """
    Test malicious uffd handler scenario.

    The page fault handler panics when receiving a page fault,
    so no events are handled and snapshot memory regions cannot be
    loaded into memory. In this case, Firecracker is designed to freeze,
    instead of silently switching to having the kernel handle page
    faults, so that it becomes obvious that something went wrong.
    """

    vm = uvm_plain
    vm.memory_monitor = None
    vm.spawn()

    # Spawn page fault handler process.
    spawn_pf_handler(vm, uffd_handler("malicious"), snapshot)

    # We expect Firecracker to freeze while resuming from a snapshot
    # due to the malicious handler's unavailability.
    try:
        with Timeout(seconds=30):
            vm.restore_from_snapshot(resume=True)
            assert False, "Firecracker should freeze"
    except (TimeoutError, requests.exceptions.ReadTimeout):
        pass
