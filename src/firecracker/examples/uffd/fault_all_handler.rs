// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provides functionality for a userspace page fault handler
//! which loads the whole region from the backing memory file
//! when a page fault occurs.

mod uffd_utils;

use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixListener;
use uffd_utils::{Runtime, UffdHandler};
use utils::time::{ClockType, get_time_us};
use crate::uffd_utils::uffd_continue;

fn main() {
    let mut args = std::env::args();
    let uffd_sock_path = args.nth(1).expect("No socket path given");
    let mem_file_path = args.next().expect("No memory file given");

    let file = File::open(mem_file_path).expect("Cannot open memfile");

    // Get Uffd from UDS. We'll use the uffd to handle PFs for Firecracker.
    let listener = UnixListener::bind(uffd_sock_path).expect("Cannot bind to socket path");
    let (stream, _) = listener.accept().expect("Cannot listen on UDS socket");

    let mut are_we_faulted_yet = false;

    let mut runtime = Runtime::new(stream, file);
    runtime.install_panic_hook();
    runtime.run(
        |uffd_handler: &mut UffdHandler| {
            let mut events = Vec::new();

            while let Some(event) = uffd_handler.read_event().unwrap() {
                events.push(event);
            }

            for event in events {
                if let userfaultfd::Event::Pagefault { addr, .. } = event {
                    if are_we_faulted_yet {
                        uffd_continue(uffd_handler.uffd.as_raw_fd(), addr as _, 4096).inspect_err(|err| eprintln!("Error during uffdio_continue: {:?}", err));
                    } else {
                        fault_all(uffd_handler);

                        are_we_faulted_yet = true;
                    }
                } else {
                    panic!("Unexpected event on userfaultfd");
                }
            }
        },
        |uffd_handler: &mut UffdHandler, offset: u64| {
            (offset, 4096)
        },
    );
}


fn fault_all(uffd_handler: &mut UffdHandler) {
    let start = get_time_us(ClockType::Monotonic);
    for region in uffd_handler.mem_regions.clone() {
        match &uffd_handler.guest_memfd {
            None => {
                uffd_handler.serve_pf(region.base_host_virt_addr as _, region.size);
            }
            Some(guest_memfd) => {
                let src = uffd_handler.backing_buffer as u64 + region.offset;
                let written = unsafe {
                    libc::pwrite64(
                        guest_memfd.as_raw_fd(),
                        src as _,
                        region.size,
                        region.offset as libc::off64_t,
                    )
                };
                assert!(written >= 0);
                let written = written as u64;

                // This code is written under the assumption that the first fault triggered by
                // firecracker is due to device restoration reading from guest memory to
                // check the virtio queues are sane. This will be reported via a uffd minor fault
                // which needs to be handled via memcpy. Importantly, we get to the uffd handler
                // with the actual guest_memfd page already faulted in, meaning pwrite will "halt"
                // once it gets to the offset of that page (e.g. written < region.size above).
                // Thus, to fault in everything, we now need to skip this one page, write the
                // remaining region, and then deal with the "gap" via uffd_handler.serve_pf().

                if (written as usize) < region.size - 4096 {
                    let src = src + written + 4096;
                    let size = region.size - written as usize - 4096;
                    let offset = region.offset + written + 4096;
                    let r = unsafe {libc::pwrite(guest_memfd.as_raw_fd(), src as _, size as _, offset as _)};
                    assert!(r > 0);
                    assert_eq!(written + r as u64, region.size as u64 - 4096);
                }

                uffd_handler.serve_pf((region.base_host_virt_addr + written) as _, 4096);
            }
        }
    }
    let end = get_time_us(ClockType::Monotonic);

    println!("Finished Faulting All: {}us", end - start);

    for region in uffd_handler.mem_regions.clone() {
        uffd_continue(uffd_handler.uffd.as_raw_fd(), region.base_host_virt_addr, region.size as u64);
    }
}