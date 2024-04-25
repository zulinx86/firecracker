#!/usr/bin/env python3
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Buildkite pipeline for custom command
"""

from common import COMMON_PARSER, BKPipeline

parser = COMMON_PARSER
parser.add_argument(
    "--command",
    required=True,
    help="command to run",
    type=str,
)
args = COMMON_PARSER.parse_args()

pipeline = BKPipeline()
pipeline.build_group(
    "⌘ Run Command",
    args.command,
)
print(pipeline.to_json())
