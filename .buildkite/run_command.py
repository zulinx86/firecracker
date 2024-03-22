#!/usr/bin/env python3
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from common import (
    COMMON_PARSER,
    pipeline_to_json,
    group,
)

parser = COMMON_PARSER
parser.add_argument(
    "--command",
    required=True,
    help="command to run",
)
args = COMMON_PARSER.parse_args()

steps = [
    group(
        "🤖 run command",
        args.command,
        args.instances,
        args.platforms,
    )
]

pipeline = {"steps": steps}
print(pipeline_to_json(pipeline))