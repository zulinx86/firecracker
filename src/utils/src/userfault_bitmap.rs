// Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines userfault bitmap data structure to be used with guest_memfd.

use std::sync::atomic::{AtomicU64, Ordering};

/// A thread-safe bitmap implementation using atomic operations.
/// The bitmap is represented as a contiguous array of `AtomicU64`,
/// where each `u64` holds 64 bits of the bitmap.
#[derive(Debug)]
pub struct UserfaultBitmap {
    bits: &'static mut [AtomicU64],
}

impl UserfaultBitmap {
    /// Creates a new `UserfaultBitmap` with the specified number of bits.
    /// All bits are initialized to 1.
    ///
    /// # Arguments
    ///
    /// * `num_bits` - The total number of bits in the bitmap. This will be rounded up to the
    ///   nearest multiple of 64 to fit complete u64 words.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::userfault_bitmap::UserfaultBitmap;
    /// let bitmap = UserfaultBitmap::new(256); // Creates a bitmap with 256 bits
    /// ```
    /* pub fn new(num_bits: usize) -> Self {
        let num_u64s = (num_bits + 63) / 64; // Round up to nearest multiple of 64
        let mut bits = Vec::with_capacity(num_u64s);
        for _ in 0..num_u64s {
            bits.push(AtomicU64::new(u64::MAX));
        }
        UserfaultBitmap {
            bits: bits.into_boxed_slice(),
        }
    } */

    /// Creates a new `UserfaultBitmap` at the specified address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The address points to valid memory that can hold num_u64s of AtomicU64
    /// - The address is properly aligned for AtomicU64
    /// - The memory will remain valid for the lifetime of the UserfaultBitmap
    /// - No other code will access this memory while the UserfaultBitmap exists
    pub unsafe fn new_at_addr(addr: *mut u8, num_bits: usize) -> Self {
        let num_u64s = (num_bits + 63) / 64; // Round up to nearest multiple of 64

        // Check alignment
        assert_eq!(
            addr.align_offset(std::mem::align_of::<AtomicU64>()),
            0,
            "Address must be aligned to AtomicU64"
        );

        // Cast the pointer
        let atomic_ptr = addr as *mut AtomicU64;

        // Create a slice from the raw parts
        let bits = unsafe { std::slice::from_raw_parts_mut(atomic_ptr, num_u64s) };

        // Initialize all bits to 1
        for atomic_u64 in bits.iter_mut() {
            atomic_u64.store(u64::MAX, Ordering::SeqCst);
        }

        UserfaultBitmap { bits }
    }

    /// Atomically sets or clears a bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to modify
    /// * `value` - The value to set: true for 1, false for 0
    ///
    /// # Example
    ///
    /// ```
    /// use utils::userfault_bitmap::UserfaultBitmap;
    /// let bitmap = UserfaultBitmap::new(64);
    /// bitmap.set(5, true); // Sets bit 5 to 1
    /// bitmap.set(5, false); // Sets bit 5 to 0
    /// ```
    pub fn set(&self, index: usize, value: bool) {
        let word_index = index / 64;
        let bit_index = index % 64;
        let mask = 1u64 << bit_index;

        if value {
            self.bits[word_index].fetch_or(mask, Ordering::SeqCst);
        } else {
            self.bits[word_index].fetch_and(!mask, Ordering::SeqCst);
        }
    }

    /// Atomically reads the value of a bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to read
    ///
    /// # Returns
    ///
    /// * `bool` - The value of the bit: true for 1, false for 0
    ///
    /// # Example
    ///
    /// ```
    /// use utils::userfault_bitmap::UserfaultBitmap;
    /// let bitmap = UserfaultBitmap::new(64);
    /// bitmap.set(5, true);
    /// assert!(bitmap.get(5));
    /// ```
    pub fn get(&self, index: usize) -> bool {
        let word_index = index / 64;
        let bit_index = index % 64;
        let mask = 1u64 << bit_index;

        (self.bits[word_index].load(Ordering::SeqCst) & mask) != 0
    }

    /// Atomically clears a range of bits starting at the specified index.
    ///
    /// This method is optimized to clear bits efficiently by operating on whole
    /// words when possible, while correctly handling partial words at the start
    /// and end of the range.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting bit index
    /// * `len` - The number of bits to clear
    ///
    /// # Example
    ///
    /// ```
    /// use utils::userfault_bitmap::UserfaultBitmap;
    /// let bitmap = UserfaultBitmap::new(128);
    /// bitmap.clear_bits(10, 70); // Clears 70 bits starting at index 10
    /// ```
    pub fn clear_bits(&self, start: usize, len: usize) {
        if len == 0 {
            return;
        }

        let start_word = start / 64;
        let end_word = (start + len - 1) / 64;
        let start_bit = start % 64;

        if start_word == end_word {
            // All bits are in the same word
            let mask = !(((!0u64) << start_bit) & ((!0u64) >> (64 - (start_bit + len))));
            self.bits[start_word].fetch_and(mask, Ordering::SeqCst);
            return;
        }

        // Handle first word
        if start_bit != 0 {
            let mask = !(!0u64 << start_bit);
            self.bits[start_word].fetch_and(mask, Ordering::SeqCst);
        } else {
            self.bits[start_word].store(0, Ordering::SeqCst);
        }

        // Clear full words
        for word_idx in (start_word + 1)..end_word {
            self.bits[word_idx].store(0, Ordering::SeqCst);
        }

        // Handle last word
        let remaining_bits = (start + len) % 64;
        if remaining_bits != 0 {
            let mask = !0u64 >> (64 - remaining_bits);
            self.bits[end_word].fetch_and(!mask, Ordering::SeqCst);
        } else {
            self.bits[end_word].store(0, Ordering::SeqCst);
        }
    }

    /// Returns a raw pointer to the underlying array of `AtomicU64`.
    ///
    /// This method is primarily intended for FFI purposes where the bitmap
    /// needs to be accessed from other languages.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the pointer is not used after the bitmap
    /// is dropped and that concurrent access is properly synchronized.
    ///
    /// # Returns
    ///
    /// * `*const AtomicU64` - A pointer to the first element of the atomic array
    pub fn as_ptr(&self) -> *const AtomicU64 {
        self.bits.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        // Test that size is rounded up properly
        let bitmap = UserfaultBitmap::new(65);
        // Should allocate 2 words for 65 bits
        assert_eq!(bitmap.bits.len(), 2);

        // Test initial state (all bits set to 1)
        for i in 0..65 {
            assert!(bitmap.get(i), "Bit {} should be set to 1", i);
        }
    }

    #[test]
    fn test_set_get() {
        let bitmap = UserfaultBitmap::new(128);

        // Test setting individual bits
        bitmap.set(0, false);
        assert!(!bitmap.get(0));
        assert!(bitmap.get(1));

        bitmap.set(63, false); // Last bit of first word
        assert!(!bitmap.get(63));
        assert!(bitmap.get(64)); // First bit of second word

        // Test setting bits across words
        bitmap.set(64, false);
        assert!(!bitmap.get(64));

        // Test toggling bits
        bitmap.set(0, true);
        assert!(bitmap.get(0));
        bitmap.set(0, false);
        assert!(!bitmap.get(0));
    }

    #[test]
    fn test_clear_bits_single_word() {
        let bitmap = UserfaultBitmap::new(64);

        // Clear bits within a single word
        bitmap.clear_bits(1, 3); // Clear bits 1, 2, 3
        assert!(bitmap.get(0));
        assert!(!bitmap.get(1));
        assert!(!bitmap.get(2));
        assert!(!bitmap.get(3));
        assert!(bitmap.get(4));
    }

    #[test]
    fn test_clear_bits_multiple_words() {
        let bitmap = UserfaultBitmap::new(128);

        // Clear bits spanning multiple words
        bitmap.clear_bits(62, 5); // Clear bits 62, 63, 64, 65, 66

        // Check bits before cleared range
        assert!(bitmap.get(61));

        // Check cleared bits
        for i in 62..67 {
            assert!(!bitmap.get(i), "Bit {} should be cleared", i);
        }

        // Check bits after cleared range
        assert!(bitmap.get(67));
    }

    #[test]
    fn test_clear_bits_edge_cases() {
        let bitmap = UserfaultBitmap::new(256);

        // Test clearing zero bits
        bitmap.clear_bits(10, 0);
        assert!(bitmap.get(10));

        // Test clearing at start
        bitmap.clear_bits(0, 3);
        assert!(!bitmap.get(0));
        assert!(!bitmap.get(1));
        assert!(!bitmap.get(2));
        assert!(bitmap.get(3));

        // Test clearing exactly one word
        bitmap.clear_bits(64, 64);
        for i in 64..128 {
            assert!(!bitmap.get(i), "Bit {} should be cleared", i);
        }
        assert!(bitmap.get(63));
        assert!(bitmap.get(128));

        // Test clearing partial words at both ends
        bitmap.clear_bits(130, 60);
        for i in 130..190 {
            assert!(!bitmap.get(i), "Bit {} should be cleared", i);
        }
        assert!(bitmap.get(129));
        assert!(bitmap.get(190));
    }

    #[test]
    fn test_as_ptr() {
        let bitmap = UserfaultBitmap::new(128);
        let ptr = bitmap.as_ptr();

        // Test that the pointer is not null
        assert!(!ptr.is_null());

        // Test that the pointer points to valid memory
        // SAFETY: Safe because the pointer is valid
        unsafe {
            let first_word = (*ptr).load(Ordering::SeqCst);
            assert_eq!(first_word, u64::MAX);
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_out_of_bounds_access() {
        let bitmap = UserfaultBitmap::new(64);
        bitmap.set(64, true); // This should panic
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let bitmap = Arc::new(UserfaultBitmap::new(128));
        let mut handles = vec![];

        // Spawn threads to clear different ranges concurrently
        for i in 0..4 {
            let bitmap_clone = bitmap.clone();
            let handle = thread::spawn(move || {
                bitmap_clone.clear_bits(i * 16, 16);
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all bits in the ranges were cleared
        for i in 0..64 {
            assert!(!bitmap.get(i), "Bit {} should be cleared", i);
        }
    }
}
