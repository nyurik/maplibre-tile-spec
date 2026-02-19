use crate::MltError;
use crate::v01::{DictionaryType, LengthType, OffsetType, PhysicalStreamType, Stream, StreamData};
use std::borrow::Cow;

/// Classified string sub-streams, used by both regular string and shared dictionary decoding.
#[derive(Default)]
struct StringStreams {
    var_binary_lengths: Option<Vec<u32>>,
    dict_lengths: Option<Vec<u32>>,
    symbol_lengths: Option<Vec<u32>>,
    data_bytes: Option<Vec<u8>>,
    dict_bytes: Option<Vec<u8>>,
    symbol_bytes: Option<Vec<u8>>,
    offsets: Option<Vec<u32>>,
}

impl StringStreams {
    fn classify(streams: Vec<Stream<'_>>) -> Result<Self, MltError> {
        use PhysicalStreamType as PST;
        let mut result = Self::default();
        for s in streams {
            match s.meta.physical_type {
                PST::Length(LengthType::VarBinary) => {
                    result.var_binary_lengths = Some(s.decode_bits_u32()?.decode_u32()?);
                }
                PST::Length(LengthType::Dictionary) => {
                    result.dict_lengths = Some(s.decode_bits_u32()?.decode_u32()?);
                }
                PST::Length(LengthType::Symbol) => {
                    result.symbol_lengths = Some(s.decode_bits_u32()?.decode_u32()?);
                }
                PST::Data(DictionaryType::None) => {
                    result.data_bytes = Some(raw_bytes(s));
                }
                PST::Data(DictionaryType::Single | DictionaryType::Shared) => {
                    result.dict_bytes = Some(raw_bytes(s));
                }
                PST::Data(DictionaryType::Fsst) => {
                    result.symbol_bytes = Some(raw_bytes(s));
                }
                PST::Offset(OffsetType::String) => {
                    result.offsets = Some(s.decode_bits_u32()?.decode_u32()?);
                }
                _ => Err(MltError::UnexpectedStreamType(s.meta.physical_type))?,
            }
        }
        Ok(result)
    }

    /// Decode dictionary entries from length + data streams, with optional FSST decompression.
    fn decode_dictionary(&self) -> Result<Vec<Cow<'_, str>>, MltError> {
        let dl = self.dict_lengths.as_deref();
        let dl = dl.ok_or(MltError::MissingStringStream("dictionary lengths"))?;
        let dd = self.dict_bytes.as_deref();
        let dd = dd.ok_or(MltError::MissingStringStream("dictionary data"))?;

        if let (Some(sym_lens), Some(sym_data)) = (&self.symbol_lengths, &self.symbol_bytes) {
            // Need to own the value because decoded FSST is a temporary buffer
            let decompressed = decode_fsst(sym_data, sym_lens, dd);
            Ok(split_to_strings(dl, &decompressed)?
                .into_iter()
                .map(|v| Cow::Owned(v.to_owned()))
                .collect())
        } else {
            Ok(split_to_strings(dl, dd)?
                .into_iter()
                .map(Into::into)
                .collect())
        }
    }
}

/// Decode string property from its sub-streams.
pub fn decode_string_streams(streams: Vec<Stream<'_>>) -> Result<Vec<String>, MltError> {
    let ss = StringStreams::classify(streams)?;

    if let Some(offsets) = &ss.offsets {
        resolve_offsets(&ss.decode_dictionary()?, offsets)
    } else if let Some(lengths) = &ss.var_binary_lengths {
        let data = ss.data_bytes.as_deref().or(ss.dict_bytes.as_deref());
        let data = data.ok_or(MltError::MissingStringStream("string data"))?;
        Ok(split_to_strings(lengths, data)?
            .into_iter()
            .map(Into::into)
            .collect())
    } else if ss.dict_lengths.is_some() {
        Ok(ss
            .decode_dictionary()?
            .into_iter()
            .map(Into::into)
            .collect())
    } else {
        Err(MltError::MissingStringStream("any usable combination"))
    }
}

/// Decode a shared dictionary from its streams, returning the dictionary entries.
pub fn decode_shared_dictionary(streams: Vec<Stream<'_>>) -> Result<Vec<String>, MltError> {
    Ok(StringStreams::classify(streams)?
        .decode_dictionary()?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Look up dictionary entries by index, converting each to an owned `String`.
pub fn resolve_offsets<S: AsRef<str>>(
    dict: &[S],
    offsets: &[u32],
) -> Result<Vec<String>, MltError> {
    offsets
        .iter()
        .map(|&idx| {
            dict.get(idx as usize)
                .map(|s| s.as_ref().to_owned())
                .ok_or(MltError::DictIndexOutOfBounds(idx, dict.len()))
        })
        .collect()
}

fn raw_bytes(s: Stream<'_>) -> Vec<u8> {
    match s.data {
        StreamData::Raw(d) => d.data.to_vec(),
        StreamData::VarInt(d) => d.data.to_vec(),
    }
}

/// Split `data` into UTF-8 strings using `lengths` as byte lengths for each entry.
fn split_to_strings<'a>(lengths: &[u32], data: &'a [u8]) -> Result<Vec<&'a str>, MltError> {
    let mut strings = Vec::with_capacity(lengths.len());
    let mut offset = 0;
    for &len in lengths {
        let len = len as usize;
        let Some(v) = data.get(offset..offset + len) else {
            return Err(MltError::BufferUnderflow {
                needed: len,
                remaining: data.len().saturating_sub(offset),
            });
        };
        strings.push(str::from_utf8(v)?);
        offset += len;
    }
    Ok(strings)
}

fn decode_fsst(symbols: &[u8], symbol_lengths: &[u32], compressed: &[u8]) -> Vec<u8> {
    // Build symbol offset table
    let mut symbol_offsets = vec![0u32; symbol_lengths.len()];
    for i in 1..symbol_lengths.len() {
        symbol_offsets[i] = symbol_offsets[i - 1] + symbol_lengths[i - 1];
    }
    let mut output = Vec::new();
    let mut i = 0;
    while i < compressed.len() {
        let sym_idx = compressed[i] as usize;
        if sym_idx == 255 {
            i += 1;
            output.push(compressed[i]);
        } else if sym_idx < symbol_lengths.len() {
            let len = symbol_lengths[sym_idx] as usize;
            let off = symbol_offsets[sym_idx] as usize;
            output.extend_from_slice(&symbols[off..off + len]);
        }
        i += 1;
    }
    output
}
