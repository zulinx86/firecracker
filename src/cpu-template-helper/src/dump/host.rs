// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum Error {}

#[derive(Serialize)]
pub struct HostCommand {
    command: String,
    stdout: Vec<String>,
    stderr: Vec<String>,
}

#[derive(Serialize)]
pub struct HostFingerprint {
    version: String,
    commands: Vec<HostCommand>,
}

pub fn dump() -> Result<HostFingerprint, Error> {
    // TODO: Add implementation to dump host fingerprint.
    Ok(HostFingerprint {
        version: crate::utils::CPU_TEMPLATE_HELPER_VERSION.to_string(),
        commands: vec![],
    })
}
