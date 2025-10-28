use log::warn;
use std::env;
use std::io::Write;
use std::process::{Command, Stdio};

/// Convert PCM f32 audio data to AAC format (in M4A container)
///
/// AAC is widely supported and provides good quality at moderate bitrates.
/// We use ffmpeg as a subprocess since there's no stable pure-Rust AAC encoder yet.
pub fn pcm_to_aac(pcm_data: &[f32], sample_rate: u32) -> Result<Vec<u8>, std::io::Error> {
    // Get configured bitrate (default 128kbps for AAC)
    let bitrate = get_configured_bitrate();

    // Convert f32 samples to i16 PCM bytes (little-endian)
    let pcm_i16: Vec<i16> = pcm_data
        .iter()
        .map(|&x| (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();

    // Convert to bytes
    let mut pcm_bytes = Vec::with_capacity(pcm_i16.len() * 2);
    for sample in pcm_i16 {
        pcm_bytes.extend_from_slice(&sample.to_le_bytes());
    }

    // Use ffmpeg to convert PCM to AAC
    // ffmpeg -f s16le -ar <sample_rate> -ac 1 -i pipe:0 -c:a aac -b:a <bitrate> -f mp4 -movflags frag_keyframe+empty_moov pipe:1
    let mut child = Command::new("ffmpeg")
        .args([
            "-f",
            "s16le", // Input format: signed 16-bit little-endian PCM
            "-ar",
            &sample_rate.to_string(), // Sample rate
            "-ac",
            "1", // Audio channels: mono
            "-i",
            "pipe:0", // Input from stdin
            "-c:a",
            "aac", // Codec: AAC
            "-b:a",
            &format!("{}k", bitrate), // Bitrate
            "-f",
            "mp4", // Output format: MP4/M4A
            "-movflags",
            "frag_keyframe+empty_moov", // Enable streaming/piping
            "pipe:1",                   // Output to stdout
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| std::io::Error::other(format!("Failed to spawn ffmpeg: {}", e)))?;

    // Write PCM data to ffmpeg stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&pcm_bytes)
            .map_err(|e| std::io::Error::other(format!("Failed to write to ffmpeg: {}", e)))?;
    }

    // Read AAC output from ffmpeg stdout
    let output = child
        .wait_with_output()
        .map_err(|e| std::io::Error::other(format!("Failed to wait for ffmpeg: {}", e)))?;

    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "ffmpeg failed with status: {}",
            output.status
        )));
    }

    Ok(output.stdout)
}

fn get_configured_bitrate() -> u32 {
    let bitrate_str = env::var("AAC_BITRATE").unwrap_or_else(|_| "128".to_string());

    match bitrate_str.parse::<u32>() {
        Ok(bitrate) if (32..=320).contains(&bitrate) => bitrate,
        _ => {
            warn!(
                "Invalid AAC_BITRATE '{}', defaulting to 128kbps",
                bitrate_str
            );
            128
        }
    }
}
