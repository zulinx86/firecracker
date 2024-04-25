#!/usr/bin/env python3
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from common import COMMON_PARSER, group, pipeline_to_json

parser = COMMON_PARSER
parser.add_argument(
    "--command",
    required=True,
    help="command to run",
    type=str,
)

args = COMMON_PARSER.parse_args()
defaults = {
    "instances": args.instances,
    "platforms": args.platforms,
}

pipeline = {
    "steps": [
        group("🔍 Run Command", args.command, **defaults)
    ]
}
print(pipeline_to_json(pipeline))
