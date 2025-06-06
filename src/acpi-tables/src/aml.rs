// Copyright © 2019 Intel Corporation
// Copyright © 2023 Rivos, Inc.
// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0

#![allow(missing_debug_implementations)]

use std::marker::PhantomData;

#[derive(Debug, Clone, thiserror::Error, displaydoc::Display)]
pub enum AmlError {
    /// Aml Path is empty
    NameEmpty,
    /// Invalid name part length
    InvalidPartLength,
    /// Invalid address range
    AddressRange,
}

pub trait Aml {
    fn append_aml_bytes(&self, _v: &mut Vec<u8>) -> Result<(), AmlError>;

    fn to_aml_bytes(&self) -> Result<Vec<u8>, AmlError> {
        let mut v = Vec::new();
        self.append_aml_bytes(&mut v)?;
        Ok(v)
    }
}

pub const ZERO: Zero = Zero {};
pub struct Zero {}

impl Aml for Zero {
    fn append_aml_bytes(&self, v: &mut Vec<u8>) -> Result<(), AmlError> {
        v.push(0u8);
        Ok(())
    }
}

pub const ONE: One = One {};
pub struct One {}

impl Aml for One {
    fn append_aml_bytes(&self, v: &mut Vec<u8>) -> Result<(), AmlError> {
        v.push(1u8);
        Ok(())
    }
}

pub const ONES: Ones = Ones {};
pub struct Ones {}

impl Aml for Ones {
    fn append_aml_bytes(&self, v: &mut Vec<u8>) -> Result<(), AmlError> {
        v.push(0xffu8);
        Ok(())
    }
}

pub struct Path {
    root: bool,
    name_parts: Vec<[u8; 4]>,
}

impl Aml for Path {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        if self.root {
            bytes.push(b'\\');
        }

        match self.name_parts.len() {
            0 => return Err(AmlError::NameEmpty),
            1 => {}
            2 => {
                bytes.push(0x2e); // DualNamePrefix
            }
            n => {
                bytes.push(0x2f); // MultiNamePrefix
                bytes.push(n.try_into().unwrap());
            }
        };

        for part in &self.name_parts {
            bytes.extend_from_slice(part);
        }

        Ok(())
    }
}

impl Path {
    pub fn new(name: &str) -> Result<Self, AmlError> {
        let root = name.starts_with('\\');
        let offset = root.into();
        let mut name_parts = Vec::new();
        for part in name[offset..].split('.') {
            if part.len() != 4 {
                return Err(AmlError::InvalidPartLength);
            }
            let mut name_part = [0u8; 4];
            name_part.copy_from_slice(part.as_bytes());
            name_parts.push(name_part);
        }

        Ok(Path { root, name_parts })
    }
}

impl TryFrom<&str> for Path {
    type Error = AmlError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Path::new(s)
    }
}

pub type Byte = u8;

impl Aml for Byte {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x0a); // BytePrefix
        bytes.push(*self);
        Ok(())
    }
}

pub type Word = u16;

impl Aml for Word {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x0b); // WordPrefix
        bytes.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

pub type DWord = u32;

impl Aml for DWord {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x0c); // DWordPrefix
        bytes.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

pub type QWord = u64;

impl Aml for QWord {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x0e); // QWordPrefix
        bytes.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

pub struct Name {
    bytes: Vec<u8>,
}

impl Aml for Name {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        // TODO: Refactor this to make more efficient but there are
        // lifetime/ownership challenges.
        bytes.extend_from_slice(&self.bytes);
        Ok(())
    }
}

impl Name {
    pub fn new(path: Path, inner: &dyn Aml) -> Result<Self, AmlError> {
        let mut bytes = vec![0x08]; // NameOp
        path.append_aml_bytes(&mut bytes)?;
        inner.append_aml_bytes(&mut bytes)?;
        Ok(Name { bytes })
    }
}

pub struct Package<'a> {
    children: Vec<&'a dyn Aml>,
}

impl Aml for Package<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = vec![self.children.len().try_into().unwrap()];
        for child in &self.children {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x12); // PackageOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

impl<'a> Package<'a> {
    pub fn new(children: Vec<&'a dyn Aml>) -> Self {
        Package { children }
    }
}

// From the ACPI spec for PkgLength:
//
// "The high 2 bits of the first byte reveal how many follow bytes are in the PkgLength. If the
// PkgLength has only one byte, bit 0 through 5 are used to encode the package length (in other
// words, values 0-63). If the package length value is more than 63, more than one byte must be
// used for the encoding in which case bit 4 and 5 of the PkgLeadByte are reserved and must be zero.
// If the multiple bytes encoding is used, bits 0-3 of the PkgLeadByte become the least significant
// 4 bits of the resulting package length value. The next ByteData will become the next least
// significant 8 bits of the resulting value and so on, up to 3 ByteData bytes. Thus, the maximum
// package length is 2**28."
//

// Also used for NamedField but in that case the length is not included in itself
fn create_pkg_length(data: &[u8], include_self: bool) -> Vec<u8> {
    let mut result = Vec::new();

    // PkgLength is inclusive and includes the length bytes
    let length_length = if data.len() < (2usize.pow(6) - 1) {
        1
    } else if data.len() < (2usize.pow(12) - 2) {
        2
    } else if data.len() < (2usize.pow(20) - 3) {
        3
    } else {
        4
    };

    let length = data.len() + if include_self { length_length } else { 0 };

    match length_length {
        1 => result.push(length.try_into().unwrap()),
        2 => {
            result.push((1u8 << 6) | TryInto::<u8>::try_into(length & 0xf).unwrap());
            result.push(TryInto::<u8>::try_into(length >> 4).unwrap())
        }
        3 => {
            result.push((2u8 << 6) | TryInto::<u8>::try_into(length & 0xf).unwrap());
            result.push(((length >> 4) & 0xff).try_into().unwrap());
            result.push(((length >> 12) & 0xff).try_into().unwrap());
        }
        _ => {
            result.push((3u8 << 6) | TryInto::<u8>::try_into(length & 0xf).unwrap());
            result.push(((length >> 4) & 0xff).try_into().unwrap());
            result.push(((length >> 12) & 0xff).try_into().unwrap());
            result.push(((length >> 20) & 0xff).try_into().unwrap());
        }
    }

    result
}

pub struct EisaName {
    value: DWord,
}

impl EisaName {
    pub fn new(name: &str) -> Result<Self, AmlError> {
        if name.len() != 7 {
            return Err(AmlError::InvalidPartLength);
        }

        let data = name.as_bytes();

        let value: u32 = ((u32::from(data[0] - 0x40) << 26)
            | (u32::from(data[1] - 0x40) << 21)
            | (u32::from(data[2] - 0x40) << 16)
            | (name.chars().nth(3).unwrap().to_digit(16).unwrap() << 12)
            | (name.chars().nth(4).unwrap().to_digit(16).unwrap() << 8)
            | (name.chars().nth(5).unwrap().to_digit(16).unwrap() << 4)
            | name.chars().nth(6).unwrap().to_digit(16).unwrap())
        .swap_bytes();

        Ok(EisaName { value })
    }
}

impl Aml for EisaName {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        self.value.append_aml_bytes(bytes)
    }
}

pub type Usize = usize;

impl Aml for Usize {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        if *self <= u8::MAX.into() {
            TryInto::<u8>::try_into(*self)
                .unwrap()
                .append_aml_bytes(bytes)
        } else if *self <= u16::MAX.into() {
            TryInto::<u16>::try_into(*self)
                .unwrap()
                .append_aml_bytes(bytes)
        } else if *self <= u32::MAX as usize {
            TryInto::<u32>::try_into(*self)
                .unwrap()
                .append_aml_bytes(bytes)
        } else {
            TryInto::<u64>::try_into(*self)
                .unwrap()
                .append_aml_bytes(bytes)
        }
    }
}

fn append_aml_string(v: &str, bytes: &mut Vec<u8>) {
    bytes.push(0x0D); // String Op
    bytes.extend_from_slice(v.as_bytes());
    bytes.push(0x0); // NullChar
}

pub type AmlStr = &'static str;

impl Aml for AmlStr {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        append_aml_string(self, bytes);
        Ok(())
    }
}

pub type AmlString = String;

impl Aml for AmlString {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        append_aml_string(self, bytes);
        Ok(())
    }
}

pub struct ResourceTemplate<'a> {
    children: Vec<&'a dyn Aml>,
}

impl Aml for ResourceTemplate<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        // Add buffer data
        for child in &self.children {
            child.append_aml_bytes(&mut tmp)?;
        }

        // Mark with end and mark checksum as as always valid
        tmp.push(0x79); // EndTag
        tmp.push(0); // zero checksum byte

        // Buffer length is an encoded integer including buffer data
        // and EndTag and checksum byte
        let mut buffer_length = tmp.len().to_aml_bytes()?;
        buffer_length.reverse();
        for byte in buffer_length {
            tmp.insert(0, byte);
        }

        // PkgLength is everything else
        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x11); // BufferOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

impl<'a> ResourceTemplate<'a> {
    pub fn new(children: Vec<&'a dyn Aml>) -> Self {
        ResourceTemplate { children }
    }
}

pub struct Memory32Fixed {
    read_write: bool, // true for read & write, false for read only
    base: u32,
    length: u32,
}

impl Memory32Fixed {
    pub fn new(read_write: bool, base: u32, length: u32) -> Self {
        Memory32Fixed {
            read_write,
            base,
            length,
        }
    }
}

impl Aml for Memory32Fixed {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x86); // Memory32Fixed
        bytes.extend_from_slice(&9u16.to_le_bytes());
        // 9 bytes of payload
        bytes.push(self.read_write.into());
        bytes.extend_from_slice(&self.base.to_le_bytes());
        bytes.extend_from_slice(&self.length.to_le_bytes());
        Ok(())
    }
}

#[derive(Copy, Clone)]
enum AddressSpaceType {
    Memory,
    Io,
    BusNumber,
}

#[derive(Copy, Clone)]
pub enum AddressSpaceCacheable {
    NotCacheable,
    Cacheable,
    WriteCombining,
    PreFetchable,
}

pub struct AddressSpace<T> {
    r#type: AddressSpaceType,
    min: T,
    max: T,
    type_flags: u8,
}

impl<T> AddressSpace<T>
where
    T: PartialOrd,
{
    pub fn new_memory(
        cacheable: AddressSpaceCacheable,
        read_write: bool,
        min: T,
        max: T,
    ) -> Result<Self, AmlError> {
        if min > max {
            return Err(AmlError::AddressRange);
        }
        Ok(AddressSpace {
            r#type: AddressSpaceType::Memory,
            min,
            max,
            type_flags: ((cacheable as u8) << 1) | u8::from(read_write),
        })
    }

    pub fn new_io(min: T, max: T) -> Result<Self, AmlError> {
        if min > max {
            return Err(AmlError::AddressRange);
        }
        Ok(AddressSpace {
            r#type: AddressSpaceType::Io,
            min,
            max,
            type_flags: 3, // EntireRange
        })
    }

    pub fn new_bus_number(min: T, max: T) -> Result<Self, AmlError> {
        if min > max {
            return Err(AmlError::AddressRange);
        }
        Ok(AddressSpace {
            r#type: AddressSpaceType::BusNumber,
            min,
            max,
            type_flags: 0,
        })
    }

    fn push_header(&self, bytes: &mut Vec<u8>, descriptor: u8, length: usize) {
        bytes.push(descriptor); // Word Address Space Descriptor
        bytes.extend_from_slice(&(TryInto::<u16>::try_into(length).unwrap()).to_le_bytes());
        bytes.push(self.r#type as u8); // type
        let generic_flags = (1 << 2) /* Min Fixed */ | (1 << 3); // Max Fixed
        bytes.push(generic_flags);
        bytes.push(self.type_flags);
    }
}

impl Aml for AddressSpace<u16> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        self.push_header(
            bytes,
            0x88,                               // Word Address Space Descriptor
            3 + 5 * std::mem::size_of::<u16>(), // 3 bytes of header + 5 u16 fields
        );

        bytes.extend_from_slice(&0u16.to_le_bytes()); // Granularity
        bytes.extend_from_slice(&self.min.to_le_bytes()); // Min
        bytes.extend_from_slice(&self.max.to_le_bytes()); // Max
        bytes.extend_from_slice(&0u16.to_le_bytes()); // Translation
        let len = self.max - self.min + 1;
        bytes.extend_from_slice(&len.to_le_bytes()); // Length
        Ok(())
    }
}

impl Aml for AddressSpace<u32> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        self.push_header(
            bytes,
            0x87,                               // DWord Address Space Descriptor
            3 + 5 * std::mem::size_of::<u32>(), // 3 bytes of header + 5 u32 fields
        );

        bytes.extend_from_slice(&0u32.to_le_bytes()); // Granularity
        bytes.extend_from_slice(&self.min.to_le_bytes()); // Min
        bytes.extend_from_slice(&self.max.to_le_bytes()); // Max
        bytes.extend_from_slice(&0u32.to_le_bytes()); // Translation
        let len = self.max - self.min + 1;
        bytes.extend_from_slice(&len.to_le_bytes()); // Length
        Ok(())
    }
}

impl Aml for AddressSpace<u64> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        self.push_header(
            bytes,
            0x8A,                               // QWord Address Space Descriptor
            3 + 5 * std::mem::size_of::<u64>(), // 3 bytes of header + 5 u64 fields
        );

        bytes.extend_from_slice(&0u64.to_le_bytes()); // Granularity
        bytes.extend_from_slice(&self.min.to_le_bytes()); // Min
        bytes.extend_from_slice(&self.max.to_le_bytes()); // Max
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Translation
        let len = self.max - self.min + 1;
        bytes.extend_from_slice(&len.to_le_bytes()); // Length
        Ok(())
    }
}

pub struct Io {
    min: u16,
    max: u16,
    alignment: u8,
    length: u8,
}

impl Io {
    pub fn new(min: u16, max: u16, alignment: u8, length: u8) -> Self {
        Io {
            min,
            max,
            alignment,
            length,
        }
    }
}

impl Aml for Io {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x47); // Io Port Descriptor
        bytes.push(1); // IODecode16
        bytes.extend_from_slice(&self.min.to_le_bytes());
        bytes.extend_from_slice(&self.max.to_le_bytes());
        bytes.push(self.alignment);
        bytes.push(self.length);
        Ok(())
    }
}

pub struct Interrupt {
    consumer: bool,
    edge_triggered: bool,
    active_low: bool,
    shared: bool,
    number: u32,
}

impl Interrupt {
    pub fn new(
        consumer: bool,
        edge_triggered: bool,
        active_low: bool,
        shared: bool,
        number: u32,
    ) -> Self {
        Interrupt {
            consumer,
            edge_triggered,
            active_low,
            shared,
            number,
        }
    }
}

impl Aml for Interrupt {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x89); // Extended IRQ Descriptor
        bytes.extend_from_slice(&6u16.to_le_bytes());
        let flags = (u8::from(self.shared) << 3)
            | (u8::from(self.active_low) << 2)
            | (u8::from(self.edge_triggered) << 1)
            | u8::from(self.consumer);
        bytes.push(flags);
        bytes.push(1u8); // count
        bytes.extend_from_slice(&self.number.to_le_bytes());
        Ok(())
    }
}

pub struct Device<'a> {
    path: Path,
    children: Vec<&'a dyn Aml>,
}

impl Aml for Device<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.path.append_aml_bytes(&mut tmp)?;

        for child in &self.children {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x82); // DeviceOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

impl<'a> Device<'a> {
    pub fn new(path: Path, children: Vec<&'a dyn Aml>) -> Self {
        Device { path, children }
    }
}

pub struct Scope<'a> {
    path: Path,
    children: Vec<&'a dyn Aml>,
}

impl Aml for Scope<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.path.append_aml_bytes(&mut tmp)?;
        for child in &self.children {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x10); // ScopeOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

impl<'a> Scope<'a> {
    pub fn new(path: Path, children: Vec<&'a dyn Aml>) -> Self {
        Scope { path, children }
    }
}

pub struct Method<'a> {
    path: Path,
    children: Vec<&'a dyn Aml>,
    args: u8,
    serialized: bool,
}

impl<'a> Method<'a> {
    pub fn new(path: Path, args: u8, serialized: bool, children: Vec<&'a dyn Aml>) -> Self {
        Method {
            path,
            children,
            args,
            serialized,
        }
    }
}

impl Aml for Method<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.path.append_aml_bytes(&mut tmp)?;
        let flags: u8 = (self.args & 0x7) | (u8::from(self.serialized) << 3);
        tmp.push(flags);
        for child in &self.children {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x14); // MethodOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

pub struct Return<'a> {
    value: &'a dyn Aml,
}

impl<'a> Return<'a> {
    pub fn new(value: &'a dyn Aml) -> Self {
        Return { value }
    }
}

impl Aml for Return<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0xa4); // ReturnOp
        self.value.append_aml_bytes(bytes)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum FieldAccessType {
    Any,
    Byte,
    Word,
    DWord,
    QWord,
    Buffer,
}

#[derive(Clone, Copy)]
pub enum FieldUpdateRule {
    Preserve = 0,
    WriteAsOnes = 1,
    WriteAsZeroes = 2,
}

pub enum FieldEntry {
    Named([u8; 4], usize),
    Reserved(usize),
}

pub struct Field {
    path: Path,

    fields: Vec<FieldEntry>,
    access_type: FieldAccessType,
    update_rule: FieldUpdateRule,
}

impl Field {
    pub fn new(
        path: Path,
        access_type: FieldAccessType,
        update_rule: FieldUpdateRule,
        fields: Vec<FieldEntry>,
    ) -> Self {
        Field {
            path,
            fields,
            access_type,
            update_rule,
        }
    }
}

impl Aml for Field {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.path.append_aml_bytes(&mut tmp)?;

        let flags: u8 = self.access_type as u8 | ((self.update_rule as u8) << 5);
        tmp.push(flags);

        for field in self.fields.iter() {
            match field {
                FieldEntry::Named(name, length) => {
                    tmp.extend_from_slice(name);
                    tmp.extend_from_slice(&create_pkg_length(&vec![0; *length], false));
                }
                FieldEntry::Reserved(length) => {
                    tmp.push(0x0);
                    tmp.extend_from_slice(&create_pkg_length(&vec![0; *length], false));
                }
            }
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x81); // FieldOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum OpRegionSpace {
    SystemMemory,
    SystemIo,
    PConfig,
    EmbeddedControl,
    Smbus,
    SystemCmos,
    PciBarTarget,
    Ipmi,
    GeneralPurposeIo,
    GenericSerialBus,
}

pub struct OpRegion {
    path: Path,
    space: OpRegionSpace,
    offset: usize,
    length: usize,
}

impl OpRegion {
    pub fn new(path: Path, space: OpRegionSpace, offset: usize, length: usize) -> Self {
        OpRegion {
            path,
            space,
            offset,
            length,
        }
    }
}

impl Aml for OpRegion {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x80); // OpRegionOp
        self.path.append_aml_bytes(bytes)?;
        bytes.push(self.space as u8);
        self.offset.append_aml_bytes(bytes)?; // RegionOffset
        self.length.append_aml_bytes(bytes)?; // RegionLen
        Ok(())
    }
}

pub struct If<'a> {
    predicate: &'a dyn Aml,
    if_children: Vec<&'a dyn Aml>,
}

impl<'a> If<'a> {
    pub fn new(predicate: &'a dyn Aml, if_children: Vec<&'a dyn Aml>) -> Self {
        If {
            predicate,
            if_children,
        }
    }
}

impl Aml for If<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.predicate.append_aml_bytes(&mut tmp)?;
        for child in self.if_children.iter() {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0xa0); // IfOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

pub struct Equal<'a> {
    left: &'a dyn Aml,
    right: &'a dyn Aml,
}

impl<'a> Equal<'a> {
    pub fn new(left: &'a dyn Aml, right: &'a dyn Aml) -> Self {
        Equal { left, right }
    }
}

impl Aml for Equal<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x93); // LEqualOp
        self.left.append_aml_bytes(bytes)?;
        self.right.append_aml_bytes(bytes)?;
        Ok(())
    }
}

pub struct LessThan<'a> {
    left: &'a dyn Aml,
    right: &'a dyn Aml,
}

impl<'a> LessThan<'a> {
    pub fn new(left: &'a dyn Aml, right: &'a dyn Aml) -> Self {
        LessThan { left, right }
    }
}

impl Aml for LessThan<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x95); // LLessOp
        self.left.append_aml_bytes(bytes)?;
        self.right.append_aml_bytes(bytes)?;
        Ok(())
    }
}

pub struct Arg(pub u8);

impl Aml for Arg {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        if self.0 > 6 {
            return Err(AmlError::InvalidPartLength);
        }
        bytes.push(0x68 + self.0); // Arg0Op
        Ok(())
    }
}

pub struct Local(pub u8);

impl Aml for Local {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        if self.0 > 7 {
            return Err(AmlError::InvalidPartLength);
        }
        bytes.push(0x60 + self.0); // Local0Op
        Ok(())
    }
}

pub struct Store<'a> {
    name: &'a dyn Aml,
    value: &'a dyn Aml,
}

impl<'a> Store<'a> {
    pub fn new(name: &'a dyn Aml, value: &'a dyn Aml) -> Self {
        Store { name, value }
    }
}

impl Aml for Store<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x70); // StoreOp
        self.value.append_aml_bytes(bytes)?;
        self.name.append_aml_bytes(bytes)?;
        Ok(())
    }
}

pub struct Mutex {
    path: Path,
    sync_level: u8,
}

impl Mutex {
    pub fn new(path: Path, sync_level: u8) -> Self {
        Self { path, sync_level }
    }
}

impl Aml for Mutex {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x01); // MutexOp
        self.path.append_aml_bytes(bytes)?;
        bytes.push(self.sync_level);
        Ok(())
    }
}

pub struct Acquire {
    mutex: Path,
    timeout: u16,
}

impl Acquire {
    pub fn new(mutex: Path, timeout: u16) -> Self {
        Acquire { mutex, timeout }
    }
}

impl Aml for Acquire {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x23); // AcquireOp
        self.mutex.append_aml_bytes(bytes)?;
        bytes.extend_from_slice(&self.timeout.to_le_bytes());
        Ok(())
    }
}

pub struct Release {
    mutex: Path,
}

impl Release {
    pub fn new(mutex: Path) -> Self {
        Release { mutex }
    }
}

impl Aml for Release {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x5b); // ExtOpPrefix
        bytes.push(0x27); // ReleaseOp
        self.mutex.append_aml_bytes(bytes)?;
        Ok(())
    }
}

pub struct Notify<'a> {
    object: &'a dyn Aml,
    value: &'a dyn Aml,
}

impl<'a> Notify<'a> {
    pub fn new(object: &'a dyn Aml, value: &'a dyn Aml) -> Self {
        Notify { object, value }
    }
}

impl Aml for Notify<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x86); // NotifyOp
        self.object.append_aml_bytes(bytes)?;
        self.value.append_aml_bytes(bytes)?;
        Ok(())
    }
}

pub struct While<'a> {
    predicate: &'a dyn Aml,
    while_children: Vec<&'a dyn Aml>,
}

impl<'a> While<'a> {
    pub fn new(predicate: &'a dyn Aml, while_children: Vec<&'a dyn Aml>) -> Self {
        While {
            predicate,
            while_children,
        }
    }
}

impl Aml for While<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.predicate.append_aml_bytes(&mut tmp)?;
        for child in self.while_children.iter() {
            child.append_aml_bytes(&mut tmp)?;
        }

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0xa2); // WhileOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

macro_rules! binary_op {
    ($name:ident, $opcode:expr) => {
        pub struct $name<'a> {
            a: &'a dyn Aml,
            b: &'a dyn Aml,
            target: &'a dyn Aml,
        }

        impl<'a> $name<'a> {
            pub fn new(target: &'a dyn Aml, a: &'a dyn Aml, b: &'a dyn Aml) -> Self {
                $name { a, b, target }
            }
        }

        impl<'a> Aml for $name<'a> {
            fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
                bytes.push($opcode); // Op for the binary operator
                self.a.append_aml_bytes(bytes)?;
                self.b.append_aml_bytes(bytes)?;
                self.target.append_aml_bytes(bytes)
            }
        }
    };
}

// binary operators: TermArg TermArg Target
binary_op!(Add, 0x72);
binary_op!(Concat, 0x73);
binary_op!(Subtract, 0x74);
binary_op!(Multiply, 0x77);
binary_op!(ShiftLeft, 0x79);
binary_op!(ShiftRight, 0x7A);
binary_op!(And, 0x7B);
binary_op!(Nand, 0x7C);
binary_op!(Or, 0x7D);
binary_op!(Nor, 0x7E);
binary_op!(Xor, 0x7F);
binary_op!(ConateRes, 0x84);
binary_op!(Mod, 0x85);
binary_op!(Index, 0x88);
binary_op!(ToString, 0x9C);

pub struct MethodCall<'a> {
    name: Path,
    args: Vec<&'a dyn Aml>,
}

impl<'a> MethodCall<'a> {
    pub fn new(name: Path, args: Vec<&'a dyn Aml>) -> Self {
        MethodCall { name, args }
    }
}

impl Aml for MethodCall<'_> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        self.name.append_aml_bytes(bytes)?;
        for arg in self.args.iter() {
            arg.append_aml_bytes(bytes)?;
        }
        Ok(())
    }
}

pub struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    pub fn new(data: Vec<u8>) -> Self {
        Buffer { data }
    }
}

impl Aml for Buffer {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        let mut tmp = Vec::new();
        self.data.len().append_aml_bytes(&mut tmp)?;
        tmp.extend_from_slice(&self.data);

        let pkg_length = create_pkg_length(&tmp, true);

        bytes.push(0x11); // BufferOp
        bytes.extend_from_slice(&pkg_length);
        bytes.extend_from_slice(&tmp);
        Ok(())
    }
}

pub struct CreateField<'a, T> {
    buffer: &'a dyn Aml,
    offset: &'a dyn Aml,
    field: Path,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> CreateField<'a, T> {
    pub fn new(buffer: &'a dyn Aml, offset: &'a dyn Aml, field: Path) -> Self {
        CreateField::<T> {
            buffer,
            offset,
            field,
            phantom: PhantomData,
        }
    }
}

impl Aml for CreateField<'_, u64> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x8f); // CreateQWordFieldOp
        self.buffer.append_aml_bytes(bytes)?;
        self.offset.append_aml_bytes(bytes)?;
        self.field.append_aml_bytes(bytes)
    }
}

impl Aml for CreateField<'_, u32> {
    fn append_aml_bytes(&self, bytes: &mut Vec<u8>) -> Result<(), AmlError> {
        bytes.push(0x8a); // CreateDWordFieldOp
        self.buffer.append_aml_bytes(bytes)?;
        self.offset.append_aml_bytes(bytes)?;
        self.field.append_aml_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device() {
        // Device (_SB.COM1)
        // {
        // Name (_HID, EisaId ("PNP0501") /* 16550A-compatible COM Serial Port */)  // _HID:
        // Hardware ID Name (_CRS, ResourceTemplate ()  // _CRS: Current Resource Settings
        // {
        // Interrupt (ResourceConsumer, Edge, ActiveHigh, Exclusive, ,, )
        // {
        // 0x00000004,
        // }
        // IO (Decode16,
        // 0x03F8,             // Range Minimum
        // 0x03F8,             // Range Maximum
        // 0x00,               // Alignment
        // 0x08,               // Length
        // )
        // }
        // }
        let com1_device = [
            0x5B, 0x82, 0x30, 0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x43, 0x4F, 0x4D, 0x31, 0x08, 0x5F,
            0x48, 0x49, 0x44, 0x0C, 0x41, 0xD0, 0x05, 0x01, 0x08, 0x5F, 0x43, 0x52, 0x53, 0x11,
            0x16, 0x0A, 0x13, 0x89, 0x06, 0x00, 0x03, 0x01, 0x04, 0x00, 0x00, 0x00, 0x47, 0x01,
            0xF8, 0x03, 0xF8, 0x03, 0x00, 0x08, 0x79, 0x00,
        ];
        assert_eq!(
            Device::new(
                "_SB_.COM1".try_into().unwrap(),
                vec![
                    &Name::new(
                        "_HID".try_into().unwrap(),
                        &EisaName::new("PNP0501").unwrap()
                    )
                    .unwrap(),
                    &Name::new(
                        "_CRS".try_into().unwrap(),
                        &ResourceTemplate::new(vec![
                            &Interrupt::new(true, true, false, false, 4),
                            &Io::new(0x3f8, 0x3f8, 0, 0x8)
                        ])
                    )
                    .unwrap()
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &com1_device[..]
        );
    }

    #[test]
    fn test_scope() {
        // Scope (_SB.MBRD)
        // {
        // Name (_CRS, ResourceTemplate ()  // _CRS: Current Resource Settings
        // {
        // Memory32Fixed (ReadWrite,
        // 0xE8000000,         // Address Base
        // 0x10000000,         // Address Length
        // )
        // })
        // }

        let mbrd_scope = [
            0x10, 0x21, 0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x4D, 0x42, 0x52, 0x44, 0x08, 0x5F, 0x43,
            0x52, 0x53, 0x11, 0x11, 0x0A, 0x0E, 0x86, 0x09, 0x00, 0x01, 0x00, 0x00, 0x00, 0xE8,
            0x00, 0x00, 0x00, 0x10, 0x79, 0x00,
        ];

        assert_eq!(
            Scope::new(
                "_SB_.MBRD".try_into().unwrap(),
                vec![
                    &Name::new(
                        "_CRS".try_into().unwrap(),
                        &ResourceTemplate::new(vec![&Memory32Fixed::new(
                            true,
                            0xE800_0000,
                            0x1000_0000
                        )])
                    )
                    .unwrap()
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &mbrd_scope[..]
        );
    }

    #[test]
    fn test_resource_template() {
        // Name (_CRS, ResourceTemplate ()  // _CRS: Current Resource Settings
        // {
        // Memory32Fixed (ReadWrite,
        // 0xE8000000,         // Address Base
        // 0x10000000,         // Address Length
        // )
        // })
        let crs_memory_32_fixed = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x11, 0x0A, 0x0E, 0x86, 0x09, 0x00, 0x01, 0x00,
            0x00, 0x00, 0xE8, 0x00, 0x00, 0x00, 0x10, 0x79, 0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![&Memory32Fixed::new(true, 0xE800_0000, 0x1000_0000)])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            crs_memory_32_fixed
        );

        // Name (_CRS, ResourceTemplate ()  // _CRS: Current Resource Settings
        // {
        // WordBusNumber (ResourceProducer, MinFixed, MaxFixed, PosDecode,
        // 0x0000,             // Granularity
        // 0x0000,             // Range Minimum
        // 0x00FF,             // Range Maximum
        // 0x0000,             // Translation Offset
        // 0x0100,             // Length
        // ,, )
        // WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
        // 0x0000,             // Granularity
        // 0x0000,             // Range Minimum
        // 0x0CF7,             // Range Maximum
        // 0x0000,             // Translation Offset
        // 0x0CF8,             // Length
        // ,, , TypeStatic, DenseTranslation)
        // WordIO (ResourceProducer, MinFixed, MaxFixed, PosDecode, EntireRange,
        // 0x0000,             // Granularity
        // 0x0D00,             // Range Minimum
        // 0xFFFF,             // Range Maximum
        // 0x0000,             // Translation Offset
        // 0xF300,             // Length
        // ,, , TypeStatic, DenseTranslation)
        // DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, Cacheable, ReadWrite,
        // 0x00000000,         // Granularity
        // 0x000A0000,         // Range Minimum
        // 0x000BFFFF,         // Range Maximum
        // 0x00000000,         // Translation Offset
        // 0x00020000,         // Length
        // ,, , AddressRangeMemory, TypeStatic)
        // DWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, NonCacheable, ReadWrite,
        // 0x00000000,         // Granularity
        // 0xC0000000,         // Range Minimum
        // 0xFEBFFFFF,         // Range Maximum
        // 0x00000000,         // Translation Offset
        // 0x3EC00000,         // Length
        // ,, , AddressRangeMemory, TypeStatic)
        // QWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, Cacheable, ReadWrite,
        // 0x0000000000000000, // Granularity
        // 0x0000000800000000, // Range Minimum
        // 0x0000000FFFFFFFFF, // Range Maximum
        // 0x0000000000000000, // Translation Offset
        // 0x0000000800000000, // Length
        // ,, , AddressRangeMemory, TypeStatic)
        // })

        // WordBusNumber from above
        let crs_word_bus_number = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x15, 0x0A, 0x12, 0x88, 0x0D, 0x00, 0x02, 0x0C,
            0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x01, 0x79, 0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![
                    &AddressSpace::new_bus_number(0x0u16, 0xffu16).unwrap(),
                ])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &crs_word_bus_number
        );

        // WordIO blocks (x 2) from above
        let crs_word_io = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x25, 0x0A, 0x22, 0x88, 0x0D, 0x00, 0x01, 0x0C,
            0x03, 0x00, 0x00, 0x00, 0x00, 0xF7, 0x0C, 0x00, 0x00, 0xF8, 0x0C, 0x88, 0x0D, 0x00,
            0x01, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x0D, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xF3, 0x79,
            0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![
                    &AddressSpace::new_io(0x0u16, 0xcf7u16).unwrap(),
                    &AddressSpace::new_io(0xd00u16, 0xffffu16).unwrap(),
                ])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &crs_word_io[..]
        );

        // DWordMemory blocks (x 2) from above
        let crs_dword_memory = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x39, 0x0A, 0x36, 0x87, 0x17, 0x00, 0x00, 0x0C,
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0xFF, 0xFF, 0x0B, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x87, 0x17, 0x00, 0x00, 0x0C, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0, 0xFF, 0xFF, 0xBF, 0xFE, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0xC0, 0x3E, 0x79, 0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![
                    &AddressSpace::new_memory(
                        AddressSpaceCacheable::Cacheable,
                        true,
                        0xa_0000u32,
                        0xb_ffffu32
                    )
                    .unwrap(),
                    &AddressSpace::new_memory(
                        AddressSpaceCacheable::NotCacheable,
                        true,
                        0xc000_0000u32,
                        0xfebf_ffffu32
                    )
                    .unwrap(),
                ])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &crs_dword_memory[..]
        );

        // QWordMemory from above
        let crs_qword_memory = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x33, 0x0A, 0x30, 0x8A, 0x2B, 0x00, 0x00, 0x0C,
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
            0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x79,
            0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![
                    &AddressSpace::new_memory(
                        AddressSpaceCacheable::Cacheable,
                        true,
                        0x8_0000_0000u64,
                        0xf_ffff_ffffu64
                    )
                    .unwrap()
                ])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &crs_qword_memory[..]
        );

        // Name (_CRS, ResourceTemplate ()  // _CRS: Current Resource Settings
        // {
        // Interrupt (ResourceConsumer, Edge, ActiveHigh, Exclusive, ,, )
        // {
        // 0x00000004,
        // }
        // IO (Decode16,
        // 0x03F8,             // Range Minimum
        // 0x03F8,             // Range Maximum
        // 0x00,               // Alignment
        // 0x08,               // Length
        // )
        // })
        //
        let interrupt_io_data = [
            0x08, 0x5F, 0x43, 0x52, 0x53, 0x11, 0x16, 0x0A, 0x13, 0x89, 0x06, 0x00, 0x03, 0x01,
            0x04, 0x00, 0x00, 0x00, 0x47, 0x01, 0xF8, 0x03, 0xF8, 0x03, 0x00, 0x08, 0x79, 0x00,
        ];

        assert_eq!(
            Name::new(
                "_CRS".try_into().unwrap(),
                &ResourceTemplate::new(vec![
                    &Interrupt::new(true, true, false, false, 4),
                    &Io::new(0x3f8, 0x3f8, 0, 0x8)
                ])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &interrupt_io_data[..]
        );
    }

    #[test]
    fn test_pkg_length() {
        assert_eq!(create_pkg_length(&[0u8; 62], true), vec![63]);
        assert_eq!(
            create_pkg_length(&[0u8; 64], true),
            vec![(1 << 6) | (66 & 0xf), 66 >> 4]
        );
        assert_eq!(
            create_pkg_length(&[0u8; 4096], true),
            vec![
                (2 << 6) | (4099 & 0xf) as u8,
                ((4099 >> 4) & 0xff).try_into().unwrap(),
                ((4099 >> 12) & 0xff).try_into().unwrap()
            ]
        );
    }

    #[test]
    fn test_package() {
        // Name (_S5, Package (0x01)  // _S5_: S5 System State
        // {
        // 0x05
        // })
        let s5_sleep_data = [0x08, 0x5F, 0x53, 0x35, 0x5F, 0x12, 0x04, 0x01, 0x0A, 0x05];

        let s5 = Name::new("_S5_".try_into().unwrap(), &Package::new(vec![&5u8])).unwrap();

        assert_eq!(s5_sleep_data.to_vec(), s5.to_aml_bytes().unwrap());
    }

    #[test]
    fn test_eisa_name() {
        assert_eq!(
            Name::new(
                "_HID".try_into().unwrap(),
                &EisaName::new("PNP0501").unwrap()
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            [0x08, 0x5F, 0x48, 0x49, 0x44, 0x0C, 0x41, 0xD0, 0x05, 0x01],
        )
    }
    #[test]
    fn test_name_path() {
        assert_eq!(
            (&"_SB_".try_into().unwrap() as &Path)
                .to_aml_bytes()
                .unwrap(),
            [0x5Fu8, 0x53, 0x42, 0x5F]
        );
        assert_eq!(
            (&"\\_SB_".try_into().unwrap() as &Path)
                .to_aml_bytes()
                .unwrap(),
            [0x5C, 0x5F, 0x53, 0x42, 0x5F]
        );
        assert_eq!(
            (&"_SB_.COM1".try_into().unwrap() as &Path)
                .to_aml_bytes()
                .unwrap(),
            [0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x43, 0x4F, 0x4D, 0x31]
        );
        assert_eq!(
            (&"_SB_.PCI0._HID".try_into().unwrap() as &Path)
                .to_aml_bytes()
                .unwrap(),
            [
                0x2F, 0x03, 0x5F, 0x53, 0x42, 0x5F, 0x50, 0x43, 0x49, 0x30, 0x5F, 0x48, 0x49, 0x44
            ]
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(128u8.to_aml_bytes().unwrap(), [0x0a, 0x80]);
        assert_eq!(1024u16.to_aml_bytes().unwrap(), [0x0b, 0x0, 0x04]);
        assert_eq!(
            (16u32 << 20).to_aml_bytes().unwrap(),
            [0x0c, 0x00, 0x00, 0x0, 0x01]
        );
        assert_eq!(
            0xdeca_fbad_deca_fbadu64.to_aml_bytes().unwrap(),
            [0x0e, 0xad, 0xfb, 0xca, 0xde, 0xad, 0xfb, 0xca, 0xde]
        );
    }

    #[test]
    fn test_name() {
        assert_eq!(
            Name::new("_SB_.PCI0._UID".try_into().unwrap(), &0x1234u16)
                .unwrap()
                .to_aml_bytes()
                .unwrap(),
            [
                0x08, // NameOp
                0x2F, // MultiNamePrefix
                0x03, // 3 name parts
                0x5F, 0x53, 0x42, 0x5F, // _SB_
                0x50, 0x43, 0x49, 0x30, // PCI0
                0x5F, 0x55, 0x49, 0x44, // _UID
                0x0b, // WordPrefix
                0x34, 0x12
            ]
        );
    }

    #[test]
    fn test_string() {
        assert_eq!(
            (&"ACPI" as &dyn Aml).to_aml_bytes().unwrap(),
            [0x0d, b'A', b'C', b'P', b'I', 0]
        );
        assert_eq!(
            "ACPI".to_owned().to_aml_bytes().unwrap(),
            [0x0d, b'A', b'C', b'P', b'I', 0]
        );
    }

    #[test]
    fn test_method() {
        assert_eq!(
            Method::new(
                "_STA".try_into().unwrap(),
                0,
                false,
                vec![&Return::new(&0xfu8)]
            )
            .to_aml_bytes()
            .unwrap(),
            [0x14, 0x09, 0x5F, 0x53, 0x54, 0x41, 0x00, 0xA4, 0x0A, 0x0F]
        );
    }

    #[test]
    fn test_field() {
        // Field (PRST, ByteAcc, NoLock, WriteAsZeros)
        // {
        // Offset (0x04),
        // CPEN,   1,
        // CINS,   1,
        // CRMV,   1,
        // CEJ0,   1,
        // Offset (0x05),
        // CCMD,   8
        // }
        //

        let field_data = [
            0x5Bu8, 0x81, 0x23, 0x50, 0x52, 0x53, 0x54, 0x41, 0x00, 0x20, 0x43, 0x50, 0x45, 0x4E,
            0x01, 0x43, 0x49, 0x4E, 0x53, 0x01, 0x43, 0x52, 0x4D, 0x56, 0x01, 0x43, 0x45, 0x4A,
            0x30, 0x01, 0x00, 0x04, 0x43, 0x43, 0x4D, 0x44, 0x08,
        ];

        assert_eq!(
            Field::new(
                "PRST".try_into().unwrap(),
                FieldAccessType::Byte,
                FieldUpdateRule::WriteAsZeroes,
                vec![
                    FieldEntry::Reserved(32),
                    FieldEntry::Named(*b"CPEN", 1),
                    FieldEntry::Named(*b"CINS", 1),
                    FieldEntry::Named(*b"CRMV", 1),
                    FieldEntry::Named(*b"CEJ0", 1),
                    FieldEntry::Reserved(4),
                    FieldEntry::Named(*b"CCMD", 8)
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &field_data[..]
        );

        // Field (PRST, DWordAcc, NoLock, Preserve)
        // {
        // CSEL,   32,
        // Offset (0x08),
        // CDAT,   32
        // }

        let field_data = [
            0x5Bu8, 0x81, 0x12, 0x50, 0x52, 0x53, 0x54, 0x03, 0x43, 0x53, 0x45, 0x4C, 0x20, 0x00,
            0x20, 0x43, 0x44, 0x41, 0x54, 0x20,
        ];

        assert_eq!(
            Field::new(
                "PRST".try_into().unwrap(),
                FieldAccessType::DWord,
                FieldUpdateRule::Preserve,
                vec![
                    FieldEntry::Named(*b"CSEL", 32),
                    FieldEntry::Reserved(32),
                    FieldEntry::Named(*b"CDAT", 32)
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &field_data[..]
        );
    }

    #[test]
    fn test_op_region() {
        // OperationRegion (PRST, SystemIo, 0x0CD8, 0x0C)
        let op_region_data = [
            0x5Bu8, 0x80, 0x50, 0x52, 0x53, 0x54, 0x01, 0x0B, 0xD8, 0x0C, 0x0A, 0x0C,
        ];

        assert_eq!(
            OpRegion::new(
                "PRST".try_into().unwrap(),
                OpRegionSpace::SystemIo,
                0xcd8,
                0xc
            )
            .to_aml_bytes()
            .unwrap(),
            &op_region_data[..]
        );
    }

    #[test]
    fn test_arg_if() {
        // Method(TEST, 1, NotSerialized) {
        // If (Arg0 == Zero) {
        // Return(One)
        // }
        // Return(Zero)
        // }
        let arg_if_data = [
            0x14, 0x0F, 0x54, 0x45, 0x53, 0x54, 0x01, 0xA0, 0x06, 0x93, 0x68, 0x00, 0xA4, 0x01,
            0xA4, 0x00,
        ];

        assert_eq!(
            Method::new(
                "TEST".try_into().unwrap(),
                1,
                false,
                vec![
                    &If::new(&Equal::new(&Arg(0), &ZERO), vec![&Return::new(&ONE)]),
                    &Return::new(&ZERO)
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &arg_if_data
        );
    }

    #[test]
    fn test_local_if() {
        // Method(TEST, 0, NotSerialized) {
        // Local0 = One
        // If (Local0 == Zero) {
        // Return(One)
        // }
        // Return(Zero)
        // }
        let local_if_data = [
            0x14, 0x12, 0x54, 0x45, 0x53, 0x54, 0x00, 0x70, 0x01, 0x60, 0xA0, 0x06, 0x93, 0x60,
            0x00, 0xA4, 0x01, 0xA4, 0x00,
        ];
        assert_eq!(
            Method::new(
                "TEST".try_into().unwrap(),
                0,
                false,
                vec![
                    &Store::new(&Local(0), &ONE),
                    &If::new(&Equal::new(&Local(0), &ZERO), vec![&Return::new(&ONE)]),
                    &Return::new(&ZERO)
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &local_if_data
        );
    }

    #[test]
    fn test_mutex() {
        // Device (_SB_.MHPC)
        // {
        // Name (_HID, EisaId("PNP0A06") /* Generic Container Device */)  // _HID: Hardware ID
        // Mutex (MLCK, 0x00)
        // Method (TEST, 0, NotSerialized)
        // {
        // Acquire (MLCK, 0xFFFF)
        // Local0 = One
        // Release (MLCK)
        // }
        // }

        let mutex_data = [
            0x5B, 0x82, 0x33, 0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x4D, 0x48, 0x50, 0x43, 0x08, 0x5F,
            0x48, 0x49, 0x44, 0x0C, 0x41, 0xD0, 0x0A, 0x06, 0x5B, 0x01, 0x4D, 0x4C, 0x43, 0x4B,
            0x00, 0x14, 0x17, 0x54, 0x45, 0x53, 0x54, 0x00, 0x5B, 0x23, 0x4D, 0x4C, 0x43, 0x4B,
            0xFF, 0xFF, 0x70, 0x01, 0x60, 0x5B, 0x27, 0x4D, 0x4C, 0x43, 0x4B,
        ];

        let mutex = Mutex::new("MLCK".try_into().unwrap(), 0);
        assert_eq!(
            Device::new(
                "_SB_.MHPC".try_into().unwrap(),
                vec![
                    &Name::new(
                        "_HID".try_into().unwrap(),
                        &EisaName::new("PNP0A06").unwrap()
                    )
                    .unwrap(),
                    &mutex,
                    &Method::new(
                        "TEST".try_into().unwrap(),
                        0,
                        false,
                        vec![
                            &Acquire::new("MLCK".try_into().unwrap(), 0xffff),
                            &Store::new(&Local(0), &ONE),
                            &Release::new("MLCK".try_into().unwrap())
                        ]
                    )
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &mutex_data[..]
        );
    }

    #[test]
    fn test_notify() {
        // Device (_SB.MHPC)
        // {
        // Name (_HID, EisaId ("PNP0A06") /* Generic Container Device */)  // _HID: Hardware ID
        // Method (TEST, 0, NotSerialized)
        // {
        // Notify (MHPC, One) // Device Check
        // }
        // }
        let notify_data = [
            0x5B, 0x82, 0x21, 0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x4D, 0x48, 0x50, 0x43, 0x08, 0x5F,
            0x48, 0x49, 0x44, 0x0C, 0x41, 0xD0, 0x0A, 0x06, 0x14, 0x0C, 0x54, 0x45, 0x53, 0x54,
            0x00, 0x86, 0x4D, 0x48, 0x50, 0x43, 0x01,
        ];

        assert_eq!(
            Device::new(
                "_SB_.MHPC".try_into().unwrap(),
                vec![
                    &Name::new(
                        "_HID".try_into().unwrap(),
                        &EisaName::new("PNP0A06").unwrap()
                    )
                    .unwrap(),
                    &Method::new(
                        "TEST".try_into().unwrap(),
                        0,
                        false,
                        vec![&Notify::new(&Path::new("MHPC").unwrap(), &ONE),]
                    )
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &notify_data[..]
        );
    }

    #[test]
    fn test_while() {
        // Device (_SB.MHPC)
        // {
        // Name (_HID, EisaId ("PNP0A06") /* Generic Container Device */)  // _HID: Hardware ID
        // Method (TEST, 0, NotSerialized)
        // {
        // Local0 = Zero
        // While ((Local0 < 0x04))
        // {
        // Local0 += One
        // }
        // }
        // }

        let while_data = [
            0x5B, 0x82, 0x28, 0x2E, 0x5F, 0x53, 0x42, 0x5F, 0x4D, 0x48, 0x50, 0x43, 0x08, 0x5F,
            0x48, 0x49, 0x44, 0x0C, 0x41, 0xD0, 0x0A, 0x06, 0x14, 0x13, 0x54, 0x45, 0x53, 0x54,
            0x00, 0x70, 0x00, 0x60, 0xA2, 0x09, 0x95, 0x60, 0x0A, 0x04, 0x72, 0x60, 0x01, 0x60,
        ];

        assert_eq!(
            Device::new(
                "_SB_.MHPC".try_into().unwrap(),
                vec![
                    &Name::new(
                        "_HID".try_into().unwrap(),
                        &EisaName::new("PNP0A06").unwrap()
                    )
                    .unwrap(),
                    &Method::new(
                        "TEST".try_into().unwrap(),
                        0,
                        false,
                        vec![
                            &Store::new(&Local(0), &ZERO),
                            &While::new(
                                &LessThan::new(&Local(0), &4usize),
                                vec![&Add::new(&Local(0), &Local(0), &ONE)]
                            )
                        ]
                    )
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &while_data[..]
        )
    }

    #[test]
    fn test_method_call() {
        // Method (TST1, 1, NotSerialized)
        // {
        // TST2 (One, One)
        // }
        //
        // Method (TST2, 2, NotSerialized)
        // {
        // TST1 (One)
        // }
        let test_data = [
            0x14, 0x0C, 0x54, 0x53, 0x54, 0x31, 0x01, 0x54, 0x53, 0x54, 0x32, 0x01, 0x01, 0x14,
            0x0B, 0x54, 0x53, 0x54, 0x32, 0x02, 0x54, 0x53, 0x54, 0x31, 0x01,
        ];

        let mut methods = Vec::new();
        methods.extend_from_slice(
            &Method::new(
                "TST1".try_into().unwrap(),
                1,
                false,
                vec![&MethodCall::new(
                    "TST2".try_into().unwrap(),
                    vec![&ONE, &ONE],
                )],
            )
            .to_aml_bytes()
            .unwrap(),
        );
        methods.extend_from_slice(
            &Method::new(
                "TST2".try_into().unwrap(),
                2,
                false,
                vec![&MethodCall::new("TST1".try_into().unwrap(), vec![&ONE])],
            )
            .to_aml_bytes()
            .unwrap(),
        );
        assert_eq!(&methods[..], &test_data[..])
    }

    #[test]
    fn test_buffer() {
        // Name (_MAT, Buffer (0x08)  // _MAT: Multiple APIC Table Entry
        // {
        // 0x00, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00   /* ........ */
        // })
        let buffer_data = [
            0x08, 0x5F, 0x4D, 0x41, 0x54, 0x11, 0x0B, 0x0A, 0x08, 0x00, 0x08, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00,
        ];

        assert_eq!(
            Name::new(
                "_MAT".try_into().unwrap(),
                &Buffer::new(vec![0x00, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00])
            )
            .unwrap()
            .to_aml_bytes()
            .unwrap(),
            &buffer_data[..]
        )
    }

    #[test]
    fn test_create_field() {
        // Method (MCRS, 0, Serialized)
        // {
        // Name (MR64, ResourceTemplate ()
        // {
        // QWordMemory (ResourceProducer, PosDecode, MinFixed, MaxFixed, Cacheable, ReadWrite,
        // 0x0000000000000000, // Granularity
        // 0x0000000000000000, // Range Minimum
        // 0xFFFFFFFFFFFFFFFE, // Range Maximum
        // 0x0000000000000000, // Translation Offset
        // 0xFFFFFFFFFFFFFFFF, // Length
        // ,, _Y00, AddressRangeMemory, TypeStatic)
        // })
        // CreateQWordField (MR64, \_SB.MHPC.MCRS._Y00._MIN, MIN)  // _MIN: Minimum Base Address
        // CreateQWordField (MR64, \_SB.MHPC.MCRS._Y00._MAX, MAX)  // _MAX: Maximum Base Address
        // CreateQWordField (MR64, \_SB.MHPC.MCRS._Y00._LEN, LEN)  // _LEN: Length
        // }
        let data = [
            0x14, 0x41, 0x06, 0x4D, 0x43, 0x52, 0x53, 0x08, 0x08, 0x4D, 0x52, 0x36, 0x34, 0x11,
            0x33, 0x0A, 0x30, 0x8A, 0x2B, 0x00, 0x00, 0x0C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x79, 0x00, 0x8F, 0x4D, 0x52, 0x36, 0x34,
            0x0A, 0x0E, 0x4D, 0x49, 0x4E, 0x5F, 0x8F, 0x4D, 0x52, 0x36, 0x34, 0x0A, 0x16, 0x4D,
            0x41, 0x58, 0x5F, 0x8F, 0x4D, 0x52, 0x36, 0x34, 0x0A, 0x26, 0x4C, 0x45, 0x4E, 0x5F,
        ];

        assert_eq!(
            Method::new(
                "MCRS".try_into().unwrap(),
                0,
                true,
                vec![
                    &Name::new(
                        "MR64".try_into().unwrap(),
                        &ResourceTemplate::new(vec![
                            &AddressSpace::new_memory(
                                AddressSpaceCacheable::Cacheable,
                                true,
                                0x0000_0000_0000_0000u64,
                                0xFFFF_FFFF_FFFF_FFFEu64
                            )
                            .unwrap()
                        ])
                    )
                    .unwrap(),
                    &CreateField::<u64>::new(
                        &Path::new("MR64").unwrap(),
                        &14usize,
                        "MIN_".try_into().unwrap()
                    ),
                    &CreateField::<u64>::new(
                        &Path::new("MR64").unwrap(),
                        &22usize,
                        "MAX_".try_into().unwrap()
                    ),
                    &CreateField::<u64>::new(
                        &Path::new("MR64").unwrap(),
                        &38usize,
                        "LEN_".try_into().unwrap()
                    ),
                ]
            )
            .to_aml_bytes()
            .unwrap(),
            &data[..]
        );
    }
}
