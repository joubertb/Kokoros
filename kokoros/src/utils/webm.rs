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
    write_ebml_master_start(output, 0x1A45DFA3)?; // EBML

    write_ebml_uint(output, 0x4286, 1)?; // EBMLVersion = 1
    write_ebml_uint(output, 0x42F7, 1)?; // EBMLReadVersion = 1
    write_ebml_uint(output, 0x42F2, 4)?; // EBMLMaxIDLength = 4
    write_ebml_uint(output, 0x42F3, 8)?; // EBMLMaxSizeLength = 8
    write_ebml_string(output, 0x4282, "webm")?; // DocType = "webm"
    write_ebml_uint(output, 0x4287, 2)?; // DocTypeVersion = 2
    write_ebml_uint(output, 0x4285, 2)?; // DocTypeReadVersion = 2

    write_ebml_master_end(output)?;

    Ok(())
}

/// Write Segment containing all media data
fn write_segment(
    output: &mut Vec<u8>,
    opus_packets: &[Vec<u8>],
    sample_rate: u32,
    duration_ns: u64,
) -> Result<(), std::io::Error> {
    write_ebml_master_start(output, 0x18538067)?; // Segment

    // Info section - metadata about the media
    write_info_section(output, duration_ns)?;

    // Tracks section - describes the audio track
    write_tracks_section(output, sample_rate)?;

    // Cluster - contains the actual audio data
    write_cluster(output, opus_packets)?;

    write_ebml_master_end(output)?;

    Ok(())
}

/// Write Info section with duration and metadata
fn write_info_section(output: &mut Vec<u8>, duration_ns: u64) -> Result<(), std::io::Error> {
    write_ebml_master_start(output, 0x1549A966)?; // Info

    // TimecodeScale: 1 million (1 tick = 1 millisecond)
    write_ebml_uint(output, 0x2AD7B1, 1_000_000)?;

    // MuxingApp and WritingApp
    write_ebml_string(output, 0x4D80, "Kokoros TTS")?; // MuxingApp
    write_ebml_string(output, 0x5741, "Kokoros TTS")?; // WritingApp

    // Duration in milliseconds
    let duration_ms = duration_ns / 1_000_000;
    write_ebml_float(output, 0x4489, duration_ms as f64)?;

    write_ebml_master_end(output)?;

    Ok(())
}

/// Write Tracks section describing the audio track
fn write_tracks_section(output: &mut Vec<u8>, sample_rate: u32) -> Result<(), std::io::Error> {
    write_ebml_master_start(output, 0x1654AE6B)?; // Tracks

    // TrackEntry
    write_ebml_master_start(output, 0xAE)?; // TrackEntry

    write_ebml_uint(output, 0xD7, 1)?; // TrackNumber = 1
    write_ebml_uint(output, 0x73C5, 1)?; // TrackUID = 1
    write_ebml_uint(output, 0x83, 2)?; // TrackType = 2 (audio)
    write_ebml_string(output, 0x86, "A_OPUS")?; // CodecID = "A_OPUS"
    write_ebml_string(output, 0x258688, "Opus")?; // CodecName = "Opus"

    // Audio settings
    write_ebml_master_start(output, 0xE1)?; // Audio

    write_ebml_float(output, 0xB5, sample_rate as f64)?; // SamplingFrequency
    write_ebml_uint(output, 0x9F, 1)?; // Channels = 1 (mono)

    write_ebml_master_end(output)?; // End Audio

    // CodecPrivate - Opus identification header
    let codec_private = create_opus_codec_private(sample_rate)?;
    write_ebml_binary(output, 0x63A2, &codec_private)?;

    write_ebml_master_end(output)?; // End TrackEntry
    write_ebml_master_end(output)?; // End Tracks

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
fn write_cluster(output: &mut Vec<u8>, opus_packets: &[Vec<u8>]) -> Result<(), std::io::Error> {
    write_ebml_master_start(output, 0x1F43B675)?; // Cluster

    write_ebml_uint(output, 0xE7, 0)?; // Timecode = 0

    // Calculate timecode increment per frame (20ms = 20 timecode units)
    let timecode_per_frame = 20u64;

    // Write each Opus packet as a SimpleBlock
    for (i, packet) in opus_packets.iter().enumerate() {
        let timecode = (i as u64) * timecode_per_frame;
        write_simple_block(output, 1, timecode as i16, packet)?;
    }

    write_ebml_master_end(output)?;

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

// EBML writing primitives

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

fn write_ebml_uint(output: &mut Vec<u8>, id: u32, value: u64) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;

    // Determine minimum bytes needed
    let bytes_needed = if value == 0 {
        1
    } else {
        ((64 - value.leading_zeros() + 7) / 8) as usize
    };

    write_ebml_size(output, bytes_needed as u64)?;

    // Write value in big-endian, only the necessary bytes
    for i in (0..bytes_needed).rev() {
        output.write_all(&[((value >> (i * 8)) & 0xFF) as u8])?;
    }

    Ok(())
}

fn write_ebml_float(output: &mut Vec<u8>, id: u32, value: f64) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    write_ebml_size(output, 8)?;
    output.write_all(&value.to_be_bytes())?;
    Ok(())
}

fn write_ebml_string(output: &mut Vec<u8>, id: u32, value: &str) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    let bytes = value.as_bytes();
    write_ebml_size(output, bytes.len() as u64)?;
    output.write_all(bytes)?;
    Ok(())
}

fn write_ebml_binary(output: &mut Vec<u8>, id: u32, data: &[u8]) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    write_ebml_size(output, data.len() as u64)?;
    output.write_all(data)?;
    Ok(())
}

fn write_ebml_master_start(output: &mut Vec<u8>, id: u32) -> Result<(), std::io::Error> {
    write_ebml_id(output, id)?;
    // Use "unknown size" marker (all 1s) - will be calculated during reading
    output.write_all(&[0xFF])?;
    Ok(())
}

fn write_ebml_master_end(_output: &mut Vec<u8>) -> Result<(), std::io::Error> {
    // Master elements with unknown size don't need end markers
    Ok(())
}

fn get_configured_bitrate() -> i32 {
    let bitrate_str = env::var("WEBM_BITRATE").unwrap_or_else(|_| "64000".to_string());

    match bitrate_str.parse::<i32>() {
        Ok(bitrate) if bitrate >= 6000 && bitrate <= 510000 => bitrate,
        _ => {
            warn!(
                "Invalid WEBM_BITRATE '{}', defaulting to 64kbps",
                bitrate_str
            );
            64000
        }
    }
}
