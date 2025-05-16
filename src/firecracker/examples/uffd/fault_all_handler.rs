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
        |uffd_handler| {
            let event = uffd_handler.read_event().unwrap().unwrap();

            if let userfaultfd::Event::Pagefault { addr, .. } = event {
                if are_we_faulted_yet {
                    _ = uffd_continue(uffd_handler.uffd.as_raw_fd(), addr as _, 4096)
                        .inspect_err(|err| println!("Error during uffdio_continue: {:?}", err));
                } else {
                    fault_all(uffd_handler, addr);

                    are_we_faulted_yet = true;
                }
            }

            uffd_handler.mem_regions.iter().map(|r| (r.offset, r.size as u64)).collect()
        },
        |_, _| {
            // TODO: understand why we're receiving these. Sending back the entire memory as prefault above
            // should have erased the kvm userfault bitmap, meaning no more kvm exits should happen.
            (0, Vec::new())
        },
    );
}


fn fault_all(uffd_handler: &mut UffdHandler, fault_addr: *mut libc::c_void) {
    let start = get_time_us(ClockType::Monotonic);
    for region in uffd_handler.mem_regions.clone() {
        match uffd_handler.guest_memfd {
            None => {
                uffd_handler.serve_pf(region.base_host_virt_addr as _, region.size);
            }
            Some(_) => {
                let written = uffd_handler.populate_via_write(region.offset as usize, region.size);

                // This code is written under the assumption that the first fault triggered by
                // firecracker is due to device restoration reading from guest memory to
                // check the virtio queues are sane. This will be reported via a uffd minor fault
                // which needs to be handled via memcpy. Importantly, we get to the uffd handler
                // with the actual guest_memfd page already faulted in, meaning pwrite will "halt"
                // once it gets to the offset of that page (e.g. written < region.size above).
                // Thus, to fault in everything, we now need to skip this one page, write the
                // remaining region, and then deal with the "gap" via uffd_handler.serve_pf().

                if written < region.size - 4096 {
                    let r = uffd_handler.populate_via_write(region.offset as usize + written + 4096, region.size - written - 4096);
                    assert_eq!(written + r, region.size - 4096);
                }
            }
        }
    }
    uffd_handler.serve_pf(fault_addr.cast(), 4096);
    let end = get_time_us(ClockType::Monotonic);

    println!("Finished Faulting All: {}us", end - start);
}