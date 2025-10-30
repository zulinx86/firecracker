#!/usr/bin/env python3
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Buildkite pipeline to test CLOCK_MONOTONIC monotonicity on restored microVMs.

1. Generate snapshots for each instance type and kernel version.
2. Wait (for instances to be terminated).
3. Restore snapshots on different instances but with the same instance type and kernel version.
"""

from common import BKPipeline

if __name__ == "__main__":
    pipeline = BKPipeline()
    per_instance = pipeline.per_instance.copy()
    per_instance.pop("instances")
    instances = [
        "m5n.metal",
        "m6i.metal",
        "m7i.metal-24xl",
        "m7i.metal-48xl",
        "m6a.metal",
        "m7a.metal-48xl",
    ]

    commands = [
        "./tools/devtool -y test --no-build --no-archive -- -m nonci integration_tests/functional/test_clock_gettime_monotonicity.py",
        "find test_results/test_clock_gettime_monotonicity_stage1 -type f -name mem |xargs -P4 -t -n1 fallocate -d",
        "mv -v test_results/test_clock_gettime_monotonicity_stage1 snapshot_artifacts",
        "mkdir -pv snapshots",
        "tar cSvf snapshots/{instance}_{kv}.tar snapshot_artifacts",
    ]
    pipeline.build_group(
        "📸 Create snapshots",
        commands,
        artifact_paths="snapshots/**/*",
        instances=instances,
    )

    pipeline.add_step("wait")

    commands = [
        "buildkite-agent artifact download snapshots/{instance}_{kv}.tar .",
        "tar xSvf snapshots/{instance}_{kv}.tar",
        "tools/devtool -y test --no-build --no-archive -- -n 42 --dist worksteal integration_tests/functional/test_clock_gettime_monotonicity.py",
    ]
    pipeline.build_group(
        "🎬 Restore snapshots and test CLOCK_MONOTONIC monotonicity",
        commands,
        instances=instances,
    )

    print(pipeline.to_json())
