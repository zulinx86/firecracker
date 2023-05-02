// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to run `{0}`: {1}")]
    RunHostCommand(String, std::io::Error),
    #[error("`{0}` exited with {1}.\nstdout: {2}\nstderr: {3}")]
    ExitCode(String, i32, String, String),
    #[error("Failed to convert command result to utf-8 string.")]
    StringFromUtf8(#[from] std::string::FromUtf8Error),
}

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
    let commands = vec![
        "uname -r",
        "dmidecode -t bios",
        "sed -n \'/^processor\\s*:\\s0/,/^$/p\' /proc/cpuinfo",
    ];

    let outputs = commands
        .iter()
        .map(|command| run_command_on_host(command))
        .collect::<Result<Vec<_>, Error>>()?;

    Ok(HostFingerprint {
        version: crate::utils::CPU_TEMPLATE_HELPER_VERSION.to_string(),
        commands: outputs,
    })
}

fn run_command_on_host(command: &str) -> Result<HostCommand, Error> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .map_err(|err| Error::RunHostCommand(command.to_string(), err))?;

    if output.status.success() {
        Ok(HostCommand {
            command: command.to_string(),
            stdout: String::from_utf8(output.stdout)?
                .lines()
                .map(|s| s.to_string())
                .collect(),
            stderr: String::from_utf8(output.stderr)?
                .lines()
                .map(|s| s.to_string())
                .collect(),
        })
    } else {
        Err(Error::ExitCode(
            command.to_string(),
            output.status.code().unwrap(),
            String::from_utf8(output.stdout)?,
            String::from_utf8(output.stderr)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_non_existing_command() {
        // Test with a non-existing command.
        // The command `sh -c unknown_command` should exit with non-zero code due to such a command
        // not found.
        let command = "unknown_command";
        match run_command_on_host(command) {
            Err(Error::ExitCode(_, _, _, _)) => (),
            Err(err) => panic!("Unexpected error: {err}"),
            Ok(_) => panic!("`{command}` should fail."),
        }
    }

    #[test]
    fn test_run_valid_command() {
        // Test with a valid command.
        let command = "ls";
        run_command_on_host(command).unwrap();
    }
}
