# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stub tests for kernel source selection; run with python3 tools/test_kernel_source.py."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps([name, str(Path.cwd()), args]) + "\n")
if name == "uname":
    print(os.environ["TEST_ARCH"])
elif name == "nproc":
    print(2)
elif name == "git":
    if "--show-toplevel" in args:
        print(os.environ["TEST_ROOT"])
    elif "--git-common-dir" in args:
        pass
    elif "clone" in args:
        Path("linux").mkdir(exist_ok=True)
    elif "tag" in args:
        print("kernel-6.18.7.amzn2023")
elif name == "make":
    if "olddefconfig" in args:
        with open(os.environ["CONFIG_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps(Path(".config").read_text()) + "\n")
    elif "distclean" not in args:
        Path("include/config").mkdir(parents=True, exist_ok=True)
        Path("include/config/kernel.release").write_text("6.18.7-prepared\n")
        for output in ("vmlinux", "arch/x86/boot/bzImage", "arch/arm64/boot/Image"):
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("kernel")
elif name == "docker" and args[0] == "images":
    print("stub-image")
"""


class KernelSourceTests(unittest.TestCase):
    """Exercise the real shell entry points with isolated source trees and tools."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kernel source ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "firecracker"
        self.root.mkdir()
        for directory in ("resources", "tools"):
            (self.root / directory).mkdir()
        for filename in ("resources/rebuild.sh", "tools/devtool", "tools/functions"):
            shutil.copy2(ROOT / filename, self.root / filename)
        shutil.copytree(
            ROOT / "resources/guest_configs", self.root / "resources/guest_configs"
        )
        patches = self.root / "resources/patches/example/6.18"
        patches.mkdir(parents=True)
        (patches / "test.patch").write_text("stub patch")
        self.source = Path(self.temp.name) / "prepared source"
        (self.source / "scripts/kconfig").mkdir(parents=True)
        (self.source / "Makefile").touch()
        (self.source / "keep-me").write_text("prepared patch")
        self.bin = Path(self.temp.name) / "bin"
        self.bin.mkdir()
        for name in (
            "git",
            "make",
            "uname",
            "nproc",
            "objcopy",
            "gzip",
            "tree",
            "docker",
        ):
            stub = self.bin / name
            stub.write_text(STUB)
            stub.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TEST_ROOT=str(self.root),
            TEST_ARCH="x86_64",
            CALL_LOG=str(self.root / "calls"),
            CONFIG_LOG=str(self.root / "configs"),
        )

    def run_script(self, *args, devtool=False):
        """Run an entry point, stubbing only dependency installation for rebuild."""
        if devtool:
            command = [
                "bash",
                str(self.root / "tools/devtool"),
                "build_ci_artifacts",
                *args,
            ]
        else:
            script = self.root / "resources/rebuild.sh"
            library = script.read_text().rsplit('main "$@"', 1)[0]
            command = [
                "bash",
                "-c",
                library
                + '\ninstall_dependencies() { echo install >> "$CALL_LOG"; }\nmain "$@"',
                str(script),
                *args,
            ]
        return subprocess.run(
            command,
            cwd=self.temp.name,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self):
        """Return structured tool invocations, excluding the install marker."""
        path = self.root / "calls"
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line != "install"
        ]

    def test_default_and_prepared_config_flow(self):
        """Both modes preserve config order; only the default prepares sources."""
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                for name in ("calls", "configs"):
                    (self.root / name).unlink(missing_ok=True)
                args = ["kernels", "6.18"]
                if prepared:
                    args += ["--kernel-source", "prepared source"]
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = self.calls()
                git_ops = [args[0] for name, _, args in calls if name == "git"]
                for operation in (
                    "clone",
                    "checkout",
                    "apply",
                    "reset",
                    "clean",
                    "--no-pager",
                ):
                    self.assertEqual(operation in git_ops, not prepared, operation)
                clean = [
                    args
                    for name, _, args in calls
                    if name == "make" and "distclean" in args
                ]
                self.assertEqual(len(clean), 0 if prepared else 2)
                configs = [
                    json.loads(line)
                    for line in (self.root / "configs").read_text().splitlines()
                ]
                config_dir = self.root / "resources/guest_configs"
                normal = [
                    "microvm-kernel-ci-x86_64-6.18.config",
                    "ci.config",
                    "nvme.config",
                ]

                def expected(names, config_dir=config_dir):
                    return "".join(
                        (config_dir / name).read_text().rstrip("\n") + "\n"
                        for name in names
                    )

                self.assertEqual(
                    configs,
                    [
                        expected(normal),
                        expected(normal + ["ftrace.config", "debug.config"]),
                    ],
                )
                targets = [
                    args[-2:]
                    for name, _, args in calls
                    if name == "make" and args[0].startswith("--jobs")
                ]
                self.assertEqual(targets, [["vmlinux", "bzImage"]] * 2)
                self.assertEqual(
                    (self.source / "keep-me").read_text(), "prepared patch"
                )

    def test_arm_debug_uses_actual_source_and_exact_output(self):
        """A stale matching artifact cannot redirect extraction away from this build."""
        self.env["TEST_ARCH"] = "aarch64"
        debug = self.root / "resources/aarch64/debug"
        debug.mkdir(parents=True)
        (debug / "vmlinux-6.18.0").write_text("stale")
        result = self.run_script("kernels", "6.18", "--kernel-source", str(self.source))
        self.assertEqual(result.returncode, 0, result.stderr)
        copies = [args for name, _, args in self.calls() if name == "objcopy"]
        artifact = str(debug / "vmlinux-6.18.7")
        self.assertEqual(
            copies[0],
            ["--only-keep-debug", str(self.source / "vmlinux"), artifact + ".debug"],
        )
        self.assertEqual(copies[1][-1], artifact)

    def test_invalid_override_fails_before_install_or_docker(self):
        """Reject missing versions, paths and unsupported combinations without work."""
        cases = [
            ("kernels", "--kernel-source", str(self.source)),
            ("kernels", "all", "--kernel-source", str(self.source)),
            ("kernels", "9.9", "--kernel-source", str(self.source)),
            ("kernels", "6.18", "--kernel-source"),
            ("kernels", "6.18", "--kernel-source", ""),
            ("kernels", "6.18", "--kernel-source", "/missing"),
            ("rootfs", "6.18", "--kernel-source", str(self.source)),
            ("kernels", "6.18", "--kernel-source", str(self.root)),
        ]
        for devtool in (False, True):
            for args in cases:
                with self.subTest(devtool=devtool, args=args):
                    (self.root / "calls").unlink(missing_ok=True)
                    result = self.run_script(*args, devtool=devtool)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("install", (self.root / "calls").read_text())
                    self.assertFalse(
                        any(name == "docker" for name, _, _ in self.calls())
                    )

    def test_devtool_mount_and_forwarding(self):
        """Forward argv without splitting spaces and keep default calls mount-free."""
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                (self.root / "calls").unlink(missing_ok=True)
                args = ["kernels", "6.18"]
                if prepared:
                    args += ["--kernel-source", "prepared source"]
                result = self.run_script(*args, devtool=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                runs = [
                    args
                    for name, _, args in self.calls()
                    if name == "docker" and args[0] == "run"
                ]
                build = runs[0]
                forwarded = build[build.index("./resources/rebuild.sh") + 1 :]
                self.assertEqual(
                    forwarded,
                    ["kernels", "6.18"]
                    + (["--kernel-source", "/kernel-source"] if prepared else []),
                )
                mount = str(self.source) + ":/kernel-source:z"
                self.assertEqual(mount in build, prepared)
                self.assertNotIn(mount, runs[1])


if __name__ == "__main__":
    unittest.main()
