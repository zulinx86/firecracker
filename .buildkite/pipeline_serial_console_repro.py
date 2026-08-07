#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the serial console login reproduction pipeline."""

import os
import re

from common import BKPipeline

TARGETS = {
    "m7g-5.10": ("m7g.metal", ("al2", "linux_5.10")),
    "m8g-24xl-6.1": ("m8g.metal-24xl", ("al2023", "linux_6.1")),
    "m8g-24xl-6.18": ("m8g.metal-24xl", ("al2023", "linux_6.18")),
    "m8g-48xl-6.18": ("m8g.metal-48xl", ("al2023", "linux_6.18")),
}
MODES = ["IMMEDIATE", "AFTER_IDLE"]


def positive_int_from_env(name, default):
    """Return a positive integer read from an environment variable."""
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def choices_from_env(name, choices):
    """Return a validated subset of choices selected through an environment variable."""
    selected = os.environ.get(name)
    if selected is None:
        return list(choices)

    selected = selected.split(",")
    invalid = set(selected) - set(choices)
    if invalid:
        raise ValueError(f"Invalid {name}: {sorted(invalid)}")
    return selected


def add_pull_request_fetch(pipeline, pull_request, revision):
    """Fetch a historical PR revision before each architecture build."""
    fetch_commands = [
        (
            "git fetch --no-tags "
            "https://github.com/firecracker-microvm/firecracker.git "
            f"refs/pull/{pull_request}/head"
        ),
        f'test "$(git rev-parse --verify FETCH_HEAD)" = "{revision}"',
    ]
    build_groups = [step for step in pipeline.steps if step.get("group") == "build"]
    if len(build_groups) != 1:
        raise RuntimeError("Expected exactly one shared build group")
    for build_step in build_groups[0]["steps"]:
        build_step["command"] = fetch_commands + build_step["command"]


revision = os.environ.get("SERIAL_REPRO_REVISION")
if revision is not None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("SERIAL_REPRO_REVISION must be a full lowercase commit SHA")
    os.environ["REVISION_A"] = revision
    os.environ["REVISION_B"] = revision

pull_request = os.environ.get("SERIAL_REPRO_PULL_REQUEST")
if pull_request is not None:
    if revision is None:
        raise ValueError("SERIAL_REPRO_PULL_REQUEST requires SERIAL_REPRO_REVISION")
    pull_request = positive_int_from_env("SERIAL_REPRO_PULL_REQUEST", 0)

repeat_count = positive_int_from_env("SERIAL_REPRO_COUNT", 100)
worker_count = positive_int_from_env("SERIAL_REPRO_WORKERS", 16)
targets = choices_from_env("SERIAL_REPRO_TARGETS", TARGETS)
modes = choices_from_env("SERIAL_REPRO_MODES", MODES)
pipeline = BKPipeline(timeout_in_minutes=45)
if pull_request is not None:
    add_pull_request_fetch(pipeline, pull_request, revision)

for mode in modes:
    BINARY_DIR = f"--binary-dir=../build/{revision} " if revision else ""
    PYTEST_OPTS = (
        f"{BINARY_DIR}-m nonci -n {worker_count} --dist worksteal "
        f"--count={repeat_count} --repeat-scope=function --maxfail=1 "
        "integration_tests/functional/test_serial_io.py "
        f"-k 'test_serial_console_login_repro and {mode}'"
    )
    for target in targets:
        instance, platform = TARGETS[target]
        pipeline.build_group(
            f"serial-console-repro-{mode.lower()}-{instance}-{platform[1]}",
            pipeline.devtool_test(pytest_opts=PYTEST_OPTS),
            instances=[instance],
            platforms=[platform],
            env={"FC_TEST_DUMP_ON_FAILURE": "1"},
        )

print(pipeline.to_json())
