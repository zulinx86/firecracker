// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This is used by `functional/test_clock_gettime.py`.
//
// This program continuously calls clock_gettime() with CLOCK_MONOTONIC and
// checks the monotonicity. If the time goes backwards, it exits with failure.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NANOSECONDS_PER_SECOND 1000000000L

void clock_gettime_wrapper(struct timespec *ts) {
        if (clock_gettime(CLOCK_MONOTONIC, ts) != 0) {
                perror("clock_gettime");
                exit(EXIT_FAILURE);
        }
}

int main(void) {
        long diff_ns;
        struct timespec prev, curr;

        // Initialize the previous time with the current time.
        clock_gettime_wrapper(&prev);

        while (1) {
                // Get the current time.
                clock_gettime_wrapper(&curr);

                // Calculate curr - prev in nanoseconds
                diff_ns = (curr.tv_sec - prev.tv_sec) * 1000000000 + (curr.tv_nsec - prev.tv_nsec);

                // Uncomment when you want to get a negative diff for testing.
                // diff_ns = (prev.tv_sec - curr.tv_sec) * 1000000000 + (prev.tv_nsec - curr.tv_nsec);

                // Print the time difference if it is negative.
                if (diff_ns < 0) {
                        printf("Negative time difference detected: %ld ns\n", diff_ns);
                        exit(EXIT_FAILURE);
                }

                // Uncomment when you want to get more verbose output.
                // printf("Time difference: %ld ns\n", diff_ns);

                // Update the previous time.
                prev = curr;
        }

        return 0;
}
