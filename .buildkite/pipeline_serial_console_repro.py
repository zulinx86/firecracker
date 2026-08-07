#!/usr/bin/env python3
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the serial console login reproduction pipeline."""

import os

from common import BKPipeline

TARGETS = [
    ("m7g.metal", ("al2", "linux_5.10")),
    ("m8g.metal-24xl", ("al2023", "linux_6.1")),
    ("m8g.metal-24xl", ("al2023", "linux_6.18")),
    ("m8g.metal-48xl", ("al2023", "linux_6.18")),
]
MODES = ["IMMEDIATE", "AFTER_IDLE"]


def positive_int_from_env(name, default):
    """Return a positive integer read from an environment variable."""
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


repeat_count = positive_int_from_env("SERIAL_REPRO_COUNT", 100)
worker_count = positive_int_from_env("SERIAL_REPRO_WORKERS", 16)
pipeline = BKPipeline(timeout_in_minutes=45)

for mode in MODES:
    PYTEST_OPTS = (
        f"-m nonci -n {worker_count} --dist worksteal "
        f"--count={repeat_count} --repeat-scope=function --maxfail=1 "
        "integration_tests/functional/test_serial_io.py "
        f"-k 'test_serial_console_login_repro and {mode}'"
    )
    for instance, platform in TARGETS:
        pipeline.build_group(
            f"serial-console-repro-{mode.lower()}-{instance}-{platform[1]}",
            pipeline.devtool_test(pytest_opts=PYTEST_OPTS),
            instances=[instance],
            platforms=[platform],
            env={"FC_TEST_DUMP_ON_FAILURE": "1"},
        )

print(pipeline.to_json())
