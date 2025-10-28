//! WebM/Matroska Audio Container Generation
//!
//! This module implements WebM audio file generation from PCM audio data.
//! WebM is a modern media container format designed for the web, based on the
//! Matroska container specification and using EBML (Extensible Binary Meta Language)
//! as its binary format.
//!
//! # WebM File Structure
//!
//! A WebM audio file consists of the following hierarchical structure:
//!
//! ```text
//! [EBML Header]           - Identifies the file as WebM
//!   - EBMLVersion: 1
//!   - DocType: "webm"
//!   - DocTypeVersion: 2
//!
//! [Segment]               - Main container for all media data
//!   [Info]                - Metadata about the file
//!     - TimecodeScale: 1ms
//!     - Duration: file length in ms
//!     - MuxingApp: "Kokoros TTS"
//!
//!   [Tracks]              - Description of audio tracks
//!     [TrackEntry]
//!       - TrackNumber: 1
//!       - TrackType: Audio
//!       - CodecID: "A_OPUS"
//!       - CodecPrivate: OpusHead identification header
//!       [Audio]
//!         - SamplingFrequency: 24000 Hz (or configured rate)
//!         - Channels: 1 (mono)
//!
//!   [Cluster(s)]          - Audio data (multiple clusters for long files)
//!     - Timecode: absolute timestamp in ms
//!     [SimpleBlock]       - Individual audio frames
//!       - TrackNumber: 1
//!       - Timecode: relative offset from cluster timecode
//!       - Flags: keyframe
//!       - Data: Opus-encoded audio packet
//! ```
//!
//! # EBML Element Format
//!
//! Every element in EBML follows this pattern:
//! ```text
//! [Element ID] [Size] [Data]
//! ```
//!
//! - **Element ID**: Variable-length (1-4 bytes) identifier
//! - **Size**: Variable-length integer indicating data size
//! - **Data**: The actual element content (can be nested elements)
//!
//! # Key Design Decisions
//!
//! ## Known vs Unknown Sizes
//! This implementation uses **known sizes** for all master elements (EBML, Segment,
//! Info, Tracks, Cluster). Known sizes improve compatibility with strict parsers
//! like Firefox's WebM decoder. Elements are buffered to calculate exact sizes
//! before writing.
//!
//! ## Multiple Clusters
//! Long audio files are split into multiple Clusters (max 1500 frames/~30 seconds each)
//! to prevent SimpleBlock timecode overflow. SimpleBlock timecodes are i16 values
//! relative to the cluster timecode, so they're limited to ±32767. Multiple clusters
//! with absolute timecodes solve this limitation.
//!
//! ## Opus Codec Integration
//! The module uses the `opus` crate to encode PCM audio into Opus packets, which
//! are then wrapped in WebM SimpleBlocks. Opus provides excellent quality at low
//! bitrates (default 64kbps) and is well-suited for speech.
//!
//! # Usage
//!
//! ```rust,ignore
//! use kokoros::utils::webm::pcm_to_webm;
//!
//! let pcm_samples: Vec<f32> = /* ... audio samples ... */;
//! let sample_rate = 24000;
//!
//! let webm_data = pcm_to_webm(&pcm_samples, sample_rate)?;
//! std::fs::write("output.webm", webm_data)?;
//! ```
//!
//! # Configuration
//!
//! - **WEBM_BITRATE**: Environment variable to set Opus encoding bitrate (6000-510000 bps)
//!   Default: 64000 (64kbps, good quality for speech)

use log::warn;
use std::env;
use std::io::Write;

/// Convert PCM f32 audio data to WebM (Matroska container with Opus codec) format
///
/// WebM is a media container format designed for the web. It uses:
/// - Matroska (EBML) container structure
/// - Opus audio codec for high-quality, low-bitrate audio
///
/// This implementation creates a proper WebM file with:
/// - EBML header identifying it as WebM
/// - Segment containing audio track metadata
/// - Clusters containing Opus-encoded audio frames
pub fn pcm_to_webm(pcm_data: &[f32], sample_rate: u32) -> Result<Vec<u8>, std::io::Error> {
    // Get configured bitrate (default 64kbps for Opus, good quality for speech)
    let bitrate = get_configured_bitrate();

    // Convert f32 samples to i16 for Opus encoder
    let pcm_i16: Vec<i16> = pcm_data
        .iter()
        .map(|&x| (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    // Create Opus encoder
    let mut encoder = opus::Encoder::new(
        sample_rate,
        opus::Channels::Mono,
        opus::Application::Voip, // Optimized for speech
    )
    .map_err(|e| std::io::Error::other(format!("Opus encoder creation failed: {:?}", e)))?;

    // Set bitrate
    encoder
        .set_bitrate(opus::Bitrate::Bits(bitrate))
        .map_err(|e| std::io::Error::other(format!("Set bitrate failed: {:?}", e)))?;

    // Opus works with frames of specific sizes (typically 20ms frames)
    // For 24kHz sample rate, 20ms = 480 samples
    let frame_size = (sample_rate as usize * 20) / 1000; // 20ms frame

    // Encode all audio frames into Opus packets
    let mut opus_packets = Vec::new();
    let mut opus_output = vec![0u8; 4000]; // Max Opus packet size

    for chunk in pcm_i16.chunks(frame_size) {
        // Pad last chunk if needed
        let mut frame = chunk.to_vec();
        if frame.len() < frame_size {
            frame.resize(frame_size, 0);
        }

        // Encode this frame
        let encoded_size = encoder
            .encode(&frame, &mut opus_output)
            .map_err(|e| std::io::Error::other(format!("Opus encoding failed: {:?}", e)))?;

        opus_packets.push(opus_output[..encoded_size].to_vec());
    }

    // Calculate audio duration in nanoseconds
    let duration_ns = (pcm_data.len() as f64 / sample_rate as f64 * 1_000_000_000.0) as u64;

    // Build WebM file
    let mut output = Vec::new();
    write_webm_file(&mut output, &opus_packets, sample_rate, duration_ns)?;

    Ok(output)
}

/// Write a complete WebM file with Opus audio
fn write_webm_file(
    output: &mut Vec<u8>,
    opus_packets: &[Vec<u8>],
    sample_rate: u32,
    duration_ns: u64,
) -> Result<(), std::io::Error> {
    // EBML Header - identifies this as a WebM file
    write_ebml_header(output)?;

    // Segment - main container for all media data
    write_segment(output, opus_packets, sample_rate, duration_ns)?;

    Ok(())
}

/// Write EBML header identifying this as WebM
fn write_ebml_header(output: &mut Vec<u8>) -> Result<(), std::io::Error> {
    let mut ebml_data = Vec::new();

    write_ebml_uint(&mut ebml_data, 0x4286, 1)?; // EBMLVersion = 1
    write_ebml_uint(&mut ebml_data, 0x42F7, 1)?; // EBMLReadVersion = 1
    write_ebml_uint(&mut ebml_data, 0x42F2, 4)?; // EBMLMaxIDLength = 4
    write_ebml_uint(&mut ebml_data, 0x42F3, 8)?; // EBMLMaxSizeLength = 8
    write_ebml_string(&mut ebml_data, 0x4282, "webm")?; // DocType = "webm"
    write_ebml_uint(&mut ebml_data, 0x4287, 2)?; // DocTypeVersion = 2
    write_ebml_uint(&mut ebml_data, 0x4285, 2)?; // DocTypeReadVersion = 2

    // Write EBML header with known size
    write_ebml_id(output, 0x1A45DFA3)?; // EBML
    write_ebml_size(output, ebml_data.len() as u64)?;
    output.write_all(&ebml_data)?;

    Ok(())
}

/// Write Segment containing all media data
fn write_segment(
    output: &mut Vec<u8>,
    opus_packets: &[Vec<u8>],
    sample_rate: u32,
    duration_ns: u64,
) -> Result<(), std::io::Error> {
    // Build segment contents first to calculate size
    let mut segment_data = Vec::new();

    // Info section - metadata about the media
    write_info_section(&mut segment_data, duration_ns)?;

    // Tracks section - describes the audio track
    write_tracks_section(&mut segment_data, sample_rate)?;

    // Cluster - contains the actual audio data
    write_cluster(&mut segment_data, opus_packets)?;

    // Write Segment header with known size
    write_ebml_id(output, 0x18538067)?; // Segment
    write_ebml_size(output, segment_data.len() as u64)?;
    output.write_all(&segment_data)?;

    Ok(())
}

/// Write Info section with duration and metadata
fn write_info_section(output: &mut Vec<u8>, duration_ns: u64) -> Result<(), std::io::Error> {
    let mut info_data = Vec::new();

    // TimecodeScale: 1 million (1 tick = 1 millisecond)
    write_ebml_uint(&mut info_data, 0x2AD7B1, 1_000_000)?;

    // MuxingApp and WritingApp
    write_ebml_string(&mut info_data, 0x4D80, "Kokoros TTS")?; // MuxingApp
    write_ebml_string(&mut info_data, 0x5741, "Kokoros TTS")?; // WritingApp

    // Duration in milliseconds
    let duration_ms = duration_ns / 1_000_000;
    write_ebml_float(&mut info_data, 0x4489, duration_ms as f64)?;

    // Write Info header with known size
    write_ebml_id(output, 0x1549A966)?; // Info
    write_ebml_size(output, info_data.len() as u64)?;
    output.write_all(&info_data)?;

    Ok(())
}

/// Write Tracks section describing the audio track
fn write_tracks_section(output: &mut Vec<u8>, sample_rate: u32) -> Result<(), std::io::Error> {
    let mut track_entry_data = Vec::new();

    write_ebml_uint(&mut track_entry_data, 0xD7, 1)?; // TrackNumber = 1
    write_ebml_uint(&mut track_entry_data, 0x73C5, 1)?; // TrackUID = 1
    write_ebml_uint(&mut track_entry_data, 0x83, 2)?; // TrackType = 2 (audio)
    write_ebml_string(&mut track_entry_data, 0x86, "A_OPUS")?; // CodecID = "A_OPUS"
    write_ebml_string(&mut track_entry_data, 0x258688, "Opus")?; // CodecName = "Opus"

    // Audio settings
    let mut audio_data = Vec::new();
    write_ebml_float(&mut audio_data, 0xB5, sample_rate as f64)?; // SamplingFrequency
    write_ebml_uint(&mut audio_data, 0x9F, 1)?; // Channels = 1 (mono)

    write_ebml_id(&mut track_entry_data, 0xE1)?; // Audio
    write_ebml_size(&mut track_entry_data, audio_data.len() as u64)?;
    track_entry_data.write_all(&audio_data)?;

    // CodecPrivate - Opus identification header
    let codec_private = create_opus_codec_private(sample_rate)?;
    write_ebml_binary(&mut track_entry_data, 0x63A2, &codec_private)?;

    // Write TrackEntry with known size
    let mut tracks_data = Vec::new();
    write_ebml_id(&mut tracks_data, 0xAE)?; // TrackEntry
    write_ebml_size(&mut tracks_data, track_entry_data.len() as u64)?;
    tracks_data.write_all(&track_entry_data)?;

    // Write Tracks header with known size
    write_ebml_id(output, 0x1654AE6B)?; // Tracks
    write_ebml_size(output, tracks_data.len() as u64)?;
    output.write_all(&tracks_data)?;

    Ok(())
}

/// Create Opus codec private data (identification header)
fn create_opus_codec_private(sample_rate: u32) -> Result<Vec<u8>, std::io::Error> {
    let mut header = Vec::new();

    // Opus identification header structure
    header.write_all(b"OpusHead")?; // Magic signature
    header.write_all(&[1])?; // Version
    header.write_all(&[1])?; // Channel count (mono)
    header.write_all(&[0x38, 0x01])?; // Pre-skip (312 samples)
    header.write_all(&sample_rate.to_le_bytes())?; // Input sample rate
    header.write_all(&[0, 0])?; // Output gain (0 dB)
    header.write_all(&[0])?; // Channel mapping family (0 = mono/stereo)

    Ok(header)
}

/// Write Cluster containing audio frames
/// For long files, splits into multiple clusters to prevent timecode overflow
fn write_cluster(output: &mut Vec<u8>, opus_packets: &[Vec<u8>]) -> Result<(), std::io::Error> {
    // Calculate timecode increment per frame (20ms = 20 timecode units)
    let timecode_per_frame = 20u64;

    // Split into clusters to prevent i16 overflow (max 1500 frames per cluster ~30 seconds)
    const MAX_FRAMES_PER_CLUSTER: usize = 1500;

    let mut global_timecode = 0u64;

    for chunk in opus_packets.chunks(MAX_FRAMES_PER_CLUSTER) {
        let mut cluster_data = Vec::new();

        // Write Cluster Timecode FIRST
        write_ebml_uint(&mut cluster_data, 0xE7, global_timecode)?; // Cluster Timecode

        // Write each Opus packet as a SimpleBlock with relative timecode
        for (i, packet) in chunk.iter().enumerate() {
            let relative_timecode = (i as u64) * timecode_per_frame;
            write_simple_block(&mut cluster_data, 1, relative_timecode as i16, packet)?;
        }

        // Write Cluster header with known size
        write_ebml_id(output, 0x1F43B675)?; // Cluster
        write_ebml_size(output, cluster_data.len() as u64)?;
        output.write_all(&cluster_data)?;

        // Update global timecode for next cluster
        global_timecode += (chunk.len() as u64) * timecode_per_frame;
    }

    Ok(())
}

/// Write a SimpleBlock containing one audio frame
fn write_simple_block(
    output: &mut Vec<u8>,
    track_number: u8,
    timecode: i16,
    data: &[u8],
) -> Result<(), std::io::Error> {
    // Calculate total block size: track(1) + timecode(2) + flags(1) + data
    let block_size = 1 + 2 + 1 + data.len();

    write_ebml_id(output, 0xA3)?; // SimpleBlock
    write_ebml_size(output, block_size as u64)?;

    output.write_all(&[track_number | 0x80])?; // Track number with continuation bit
    output.write_all(&timecode.to_be_bytes())?; // Timecode (big-endian)
    output.write_all(&[0x80])?; // Flags: keyframe
    output.write_all(data)?; // Opus packet data

    Ok(())
}

// ============================================================================
// EBML Writing Primitives
// ============================================================================
//
// EBML (Extensible Binary Meta Language) is the binary format used by WebM/Matroska.
// All elements in EBML follow the pattern: [Element ID] [Size] [Data]
//
// These functions provide low-level primitives for writing EBML structures.
// They handle variable-size integer encoding, which allows efficient representation
// of both small and large values.

/// Write an EBML element ID
///
/// Element IDs uniquely identify each type of element in the WebM/Matroska format.
/// IDs are variable-length (1-4 bytes) and written in big-endian format.
///
/// # Examples of Element IDs
/// - `0x1A45DFA3` - EBML header (4 bytes)
/// - `0x1F43B675` - Cluster (4 bytes)
/// - `0xE7` - Timecode (1 byte)
/// - `0xA3` - SimpleBlock (1 byte)
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `id` - The element ID as a 32-bit integer
fn write_ebml_id(output: &mut Vec<u8>, id: u32) -> Result<(), std::io::Error> {
    if id <= 0xFF {
        output.write_all(&[id as u8])?;
    } else if id <= 0xFFFF {
        output.write_all(&(id as u16).to_be_bytes())?;
    } else if id <= 0xFFFFFF {
        output.write_all(&[((id >> 16) & 0xFF) as u8])?;
        output.write_all(&((id & 0xFFFF) as u16).to_be_bytes())?;
    } else {
        output.write_all(&id.to_be_bytes())?;
    }
    Ok(())
}

/// Write an EBML element size using variable-size integer encoding
///
/// EBML uses a special variable-length encoding for sizes where the first byte
/// indicates how many total bytes are used. This allows efficient representation:
/// - 0-126 bytes: 1 byte (format: 1xxxxxxx)
/// - 127-16383 bytes: 2 bytes (format: 01xxxxxx xxxxxxxx)
/// - 16384-2097151 bytes: 3 bytes (format: 001xxxxx xxxxxxxx xxxxxxxx)
/// - Larger sizes: 4+ bytes
///
/// The leading 1-bits indicate the total length, and the remaining bits store the value.
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `size` - The size value to encode (0 to 2^56-1)
///
/// # Example
/// - Size 100 → [0x84, 0x64] (1-byte marker, value fits in 7 bits)
/// - Size 5000 → [0x53, 0x88] (2-byte marker, value fits in 14 bits)
fn write_ebml_size(output: &mut Vec<u8>, size: u64) -> Result<(), std::io::Error> {
    // EBML variable-size integer encoding
    if size < 0x7F {
        output.write_all(&[(0x80 | size) as u8])?;
    } else if size < 0x3FFF {
        output.write_all(&[0x40 | ((size >> 8) & 0x3F) as u8, (size & 0xFF) as u8])?;
    } else if size < 0x1FFFFF {
        output.write_all(&[
            0x20 | ((size >> 16) & 0x1F) as u8,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ])?;
    } else {
        // For simplicity, use 4-byte size for larger values
        output.write_all(&[
            0x10 | ((size >> 24) & 0x0F) as u8,
            ((size >> 16) & 0xFF) as u8,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ])?;
    }
    Ok(())
}

/// Write an EBML unsigned integer element
///
/// Writes a complete EBML element containing an unsigned integer value.
/// The integer is stored in big-endian format using the minimum number of bytes needed.
///
/// Format: [Element ID] [Size] [Value in big-endian]
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `id` - The element ID (e.g., 0xD7 for TrackNumber)
/// * `value` - The unsigned integer value to write
///
/// # Examples
/// - TrackNumber=1: `write_ebml_uint(output, 0xD7, 1)` → [0xD7, 0x81, 0x01]
/// - SampleRate=24000: `write_ebml_uint(output, 0xB5, 24000)` → [0xB5, 0x83, 0x00, 0x5D, 0xC0]
fn write_ebml_uint(output: &mut Vec<u8>, id: u32, value: u64) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;

    // Determine minimum bytes needed
    let bytes_needed = if value == 0 {
        1
    } else {
        (64 - value.leading_zeros()).div_ceil(8) as usize
    };

    write_ebml_size(output, bytes_needed as u64)?;

    // Write value in big-endian, only the necessary bytes
    for i in (0..bytes_needed).rev() {
        output.write_all(&[((value >> (i * 8)) & 0xFF) as u8])?;
    }

    Ok(())
}

/// Write an EBML floating-point element
///
/// Writes a complete EBML element containing a 64-bit floating-point value.
/// Always uses 8 bytes (IEEE 754 double precision) in big-endian format.
///
/// Format: [Element ID] [Size=8] [IEEE 754 double in big-endian]
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `id` - The element ID (e.g., 0x4489 for Duration, 0xB5 for SamplingFrequency)
/// * `value` - The floating-point value to write
///
/// # Examples
/// - Duration=1530ms: `write_ebml_float(output, 0x4489, 1530.0)`
/// - SamplingFrequency=24000Hz: `write_ebml_float(output, 0xB5, 24000.0)`
fn write_ebml_float(output: &mut Vec<u8>, id: u32, value: f64) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    write_ebml_size(output, 8)?;
    output.write_all(&value.to_be_bytes())?;
    Ok(())
}

/// Write an EBML string element
///
/// Writes a complete EBML element containing a UTF-8 string value.
/// The string is written as raw UTF-8 bytes without null termination.
///
/// Format: [Element ID] [Size] [UTF-8 bytes]
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `id` - The element ID (e.g., 0x4282 for DocType, 0x86 for CodecID)
/// * `value` - The string value to write
///
/// # Examples
/// - DocType="webm": `write_ebml_string(output, 0x4282, "webm")`
/// - CodecID="A_OPUS": `write_ebml_string(output, 0x86, "A_OPUS")`
fn write_ebml_string(output: &mut Vec<u8>, id: u32, value: &str) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    let bytes = value.as_bytes();
    write_ebml_size(output, bytes.len() as u64)?;
    output.write_all(bytes)?;
    Ok(())
}

/// Write an EBML binary element
///
/// Writes a complete EBML element containing arbitrary binary data.
/// Used for codec-specific data, raw byte sequences, etc.
///
/// Format: [Element ID] [Size] [Raw bytes]
///
/// # Arguments
/// * `output` - The buffer to write to
/// * `id` - The element ID (e.g., 0x63A2 for CodecPrivate)
/// * `data` - The binary data to write
///
/// # Examples
/// - CodecPrivate (OpusHead): `write_ebml_binary(output, 0x63A2, opus_header_bytes)`
fn write_ebml_binary(output: &mut Vec<u8>, id: u32, data: &[u8]) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    write_ebml_size(output, data.len() as u64)?;
    output.write_all(data)?;
    Ok(())
}

/// Get configured WebM bitrate from environment variable
///
/// Reads the WEBM_BITRATE environment variable to determine Opus encoding bitrate.
/// Falls back to 64kbps if not set or invalid.
///
/// # Valid Range
/// 6000 - 510000 bits per second (6kbps - 510kbps)
///
/// # Default
/// 64000 bits per second (64kbps) - good quality for speech
///
/// # Returns
/// The bitrate in bits per second as an i32
fn get_configured_bitrate() -> i32 {
    let bitrate_str = env::var("WEBM_BITRATE").unwrap_or_else(|_| "64000".to_string());

    match bitrate_str.parse::<i32>() {
        Ok(bitrate) if (6000..=510000).contains(&bitrate) => bitrate,
        _ => {
            warn!(
                "Invalid WEBM_BITRATE '{}', defaulting to 64kbps",
                bitrate_str
            );
            64000
        }
    }
}
