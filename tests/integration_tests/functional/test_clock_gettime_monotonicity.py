# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that CLOCK_MONOTONIC is indeed monotonic inside the microVM."""

import os
import re
from pathlib import Path

import pytest

import host_tools.cargo_build as build_tools
from conftest import uvm_booted
from framework.defs import FC_WORKSPACE_DIR
from framework.properties import global_props

CLOCK_GETTIME_MONOTONICITY_BIN = "clock_gettime_monotonicity"
CLOCK_GETTIME_MONOTONICITY_SRC = f"host_tools/{CLOCK_GETTIME_MONOTONICITY_BIN}.c"
CLOCK_GETTIME_MONOTONICITY_REMOTE_PATH = f"/tmp/{CLOCK_GETTIME_MONOTONICITY_BIN}"

VCPU_COUNT = 4
MEM_SIZE_MIB = 512


if global_props.cpu_architecture != "x86_64":
    pytest.skip(
        "This test is relevant for TSC clocksource, which is only available on x86_64.",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def clock_gettime_monotonicity_bin(test_fc_session_root_path):
    """Build a binary that tests clock_gettime monotonicity."""
    bin_path = os.path.join(test_fc_session_root_path, CLOCK_GETTIME_MONOTONICITY_BIN)
    build_tools.gcc_compile(
        CLOCK_GETTIME_MONOTONICITY_SRC,
        bin_path,
    )
    yield bin_path


@pytest.fixture
def uvm_booted_custom(
    microvm_factory,
    guest_kernel,
    rootfs,
    clock_gettime_monotonicity_bin,
):
    """Custom fixture for booted microVM."""
    uvm = uvm_booted(
        microvm_factory,
        guest_kernel,
        rootfs,
        cpu_template=None,
        pci_enabled=False,
        vcpu_count=VCPU_COUNT,
        mem_size_mib=MEM_SIZE_MIB,
    )
    uvm.ssh.scp_put(
        clock_gettime_monotonicity_bin,
        CLOCK_GETTIME_MONOTONICITY_REMOTE_PATH,
    )
    return uvm


@pytest.fixture
def uvm_snapshot(uvm_booted_custom):
    """Fixture for snapshot."""
    uvm = uvm_booted_custom
    snapshot = uvm.snapshot_full()
    uvm.kill()
    return snapshot


@pytest.fixture(params=[None, "on_demand", "slow"])
def uffd_handler(request):
    """Fixture for UFFD."""
    return request.param


@pytest.fixture
def uvm_restored_custom(
    microvm_factory,
    uvm_snapshot,
    uffd_handler,
):
    """Custom fixture for restored microVM."""
    if uffd_handler is None:
        uvm = microvm_factory.build_from_snapshot(uvm_snapshot)
    else:
        uvm = microvm_factory.build()
        uvm.spawn()
        uvm.restore_from_snapshot(
            uvm_snapshot, resume=True, uffd_handler_name=uffd_handler
        )
    return uvm


@pytest.fixture(params=["tsc", "kvm-clock"])
def clocksource(request):
    """Fixture for clocksource"""
    return request.param


def check_clock_gettime_monotonicity(uvm, clocksource):
    """Common part for test_clock_gettime_monotonicity_*()"""
    # Set clocksource
    cmd = f"echo {clocksource} > /sys/devices/system/clocksource/clocksource0/current_clocksource"
    _, stdout, stderr = uvm.ssh.check_output(cmd)

    # Run the clock_gettime binary with a timeout of 30m.
    cmd = f"timeout 30m {CLOCK_GETTIME_MONOTONICITY_REMOTE_PATH}"
    code, stdout, stderr = uvm.ssh.run(cmd, timeout=None)
    assert (
        code == 124
    ), f"clock_gettime exited with non-124 code ({code}):\nstdout:{stdout}\nstderr:\n{stderr}\n"


def test_clock_gettime_monotonicity_booted(uvm_booted_custom, clocksource):
    """Test that CLOCK_MONOTONIC is indeed monotonic inside the booted microVM."""
    check_clock_gettime_monotonicity(uvm_booted_custom, clocksource)


def test_clock_gettime_monotonicity_restored(uvm_restored_custom, clocksource):
    """Test that CLOCK_MONOTONIC is indeed monotonic inside the restored microVM."""
    check_clock_gettime_monotonicity(uvm_restored_custom, clocksource)


@pytest.mark.nonci
def test_clock_gettime_monotonicity_stage1(uvm_booted_custom, results_dir):
    """Stage 1"""
    vm = uvm_booted_custom
    snapshot = vm.snapshot_full()
    snapshot.save_to(results_dir)


def get_snapshot_dirs():
    """Get all the snapshot directories"""
    snapshot_root_dir = Path(FC_WORKSPACE_DIR) / "snapshot_artifacts"
    if not snapshot_root_dir.exists():
        return []
    for snapshot_dir in snapshot_root_dir.iterdir():
        assert snapshot_dir.is_dir()
        yield pytest.param(snapshot_dir, id=snapshot_dir.name)


@pytest.fixture(params=get_snapshot_dirs())
def snapshot_dir(request):
    """Fixture for snapshot directory."""
    return request.param


@pytest.fixture
def uvm_multistage_restored_custom(
    microvm_factory,
    snapshot_dir,
    uffd_handler,
):
    """Custom fixture for multi-stage restored microVM."""
    uvm = microvm_factory.build()
    uvm.spawn()
    uvm.restore_from_path(snapshot_dir, resume=True, uffd_handler_name=uffd_handler)
    return uvm


def test_clock_gettime_monotonicity_stage2(uvm_multistage_restored_custom, clocksource):
    """Stage 2"""
    uvm = uvm_multistage_restored_custom
    check_clock_gettime_monotonicity(uvm, clocksource)
