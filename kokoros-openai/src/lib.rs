use std::error::Error;
use std::io::{self};

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, routing::get, routing::post};
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::aac::pcm_to_aac,
    utils::mp3::pcm_to_mp3,
    utils::wav::{WavHeader, write_audio_chunk},
    utils::webm::pcm_to_webm,
};
use log::debug;
use serde::Deserialize;
use tower_http::cors::CorsLayer;

#[derive(Deserialize, Default, Debug)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Wav,
    Mp3,
    Webm,
    Aac,
}

#[derive(Deserialize)]
struct Voice(String);

impl Default for Voice {
    fn default() -> Self {
        Self("af_sky".into())
    }
}

#[derive(Deserialize)]
struct Speed(f32);

impl Default for Speed {
    fn default() -> Self {
        Self(1.)
    }
}

#[derive(Deserialize)]
struct SpeechRequest {
    // Only one Kokoro model exists
    #[allow(dead_code)]
    model: String,

    /// Input text to synthesize. Supports pause tags for mid-text pauses:
    /// - `<pause>` - Insert default pause (500ms = 0.5 seconds)
    /// - `<pause:N>` - Insert custom pause with N milliseconds (e.g., `<pause:1000>` for 1 second)
    ///
    /// Examples:
    /// - "Hello there. <pause> How are you?"  // 500ms pause
    /// - "First part. <pause:2000> After a 2 second pause."
    /// - "A <pause:250> B <pause:500> C <pause:1000> D"
    input: String,

    #[serde(default)]
    voice: Voice,

    // Must be WAV
    #[allow(dead_code)]
    #[serde(default)]
    response_format: AudioFormat,

    #[serde(default)]
    speed: Speed,

    /// Optional silence tokens to add at the beginning of the audio.
    /// Note: Pause tags within the text are handled separately.
    #[serde(default)]
    initial_silence: Option<usize>,
}

/// Request for the /v1/audio/pcm endpoint.
/// Same as SpeechRequest but without response_format (always returns raw PCM).
#[derive(Deserialize)]
struct PcmRequest {
    #[allow(dead_code)]
    model: String,

    input: String,

    #[serde(default)]
    voice: Voice,

    #[serde(default)]
    speed: Speed,

    #[serde(default)]
    initial_silence: Option<usize>,
}

pub async fn create_server(tts: TTSKoko) -> Router {
    debug!("create_server()");

    Router::new()
        .route("/", get(handle_home))
        .route("/v1/audio/speech", post(handle_tts))
        .route("/v1/audio/pcm", post(handle_pcm))
        .layer(CorsLayer::permissive())
        .with_state(tts)
}

pub use axum::serve;

#[derive(Debug)]
enum SpeechError {
    // Deciding to modify this example in order to see errors
    // (e.g. with tracing) is up to the developer
    #[allow(dead_code)]
    Koko(Box<dyn Error>),

    #[allow(dead_code)]
    Header(io::Error),

    #[allow(dead_code)]
    Chunk(io::Error),

    #[allow(dead_code)]
    Mp3Conversion(std::io::Error),
}

impl IntoResponse for SpeechError {
    fn into_response(self) -> Response {
        // None of these errors make sense to expose to the user of the API
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    }
}

/// Returns a 200 OK response to make it easier to check if the server is
/// running.
async fn handle_home() -> &'static str {
    "OK"
}

async fn handle_tts(
    State(tts): State<TTSKoko>,
    Json(SpeechRequest {
        model: _,
        input,
        voice: Voice(voice),
        response_format,
        speed: Speed(speed),
        initial_silence,
    }): Json<SpeechRequest>,
) -> Result<Response, SpeechError> {
    debug!(
        "Processing TTS request for voice: {}, speed: {}, text length: {}",
        voice,
        speed,
        input.len()
    );

    let raw_audio = tts
        .tts_raw_audio(&input, "en-us", &voice, speed, initial_silence)
        .map_err(SpeechError::Koko)?;

    let sample_rate = TTSKokoInitConfig::default().sample_rate;

    let (content_type, audio_data) = match response_format {
        AudioFormat::Wav => {
            let mut wav_data = Vec::default();
            let header = WavHeader::new(1, sample_rate, 32);
            header
                .write_header(&mut wav_data)
                .map_err(SpeechError::Header)?;
            write_audio_chunk(&mut wav_data, &raw_audio).map_err(SpeechError::Chunk)?;

            ("audio/wav", wav_data)
        }
        AudioFormat::Mp3 => {
            let mp3_data =
                pcm_to_mp3(&raw_audio, sample_rate).map_err(SpeechError::Mp3Conversion)?;

            ("audio/mpeg", mp3_data)
        }
        AudioFormat::Webm => {
            let webm_data =
                pcm_to_webm(&raw_audio, sample_rate).map_err(SpeechError::Mp3Conversion)?;

            ("audio/webm", webm_data)
        }
        AudioFormat::Aac => {
            let aac_data =
                pcm_to_aac(&raw_audio, sample_rate).map_err(SpeechError::Mp3Conversion)?;

            ("audio/mp4", aac_data)
        }
    };

    Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(audio_data.into())
        .map_err(|e| SpeechError::Mp3Conversion(std::io::Error::other(e)))
}

/// Returns raw PCM f32le audio data with metadata in response headers.
/// This avoids encoding overhead — the caller (worker) encodes to the
/// final format(s) it needs using PyAV.
async fn handle_pcm(
    State(tts): State<TTSKoko>,
    Json(PcmRequest {
        model: _,
        input,
        voice: Voice(voice),
        speed: Speed(speed),
        initial_silence,
    }): Json<PcmRequest>,
) -> Result<Response, SpeechError> {
    debug!(
        "Processing PCM request for voice: {}, speed: {}, text length: {}",
        voice,
        speed,
        input.len()
    );

    let raw_audio = tts
        .tts_raw_audio(&input, "en-us", &voice, speed, initial_silence)
        .map_err(SpeechError::Koko)?;

    let sample_rate = TTSKokoInitConfig::default().sample_rate;

    // Convert f32 samples to little-endian bytes
    let mut pcm_bytes = Vec::with_capacity(raw_audio.len() * 4);
    for sample in &raw_audio {
        pcm_bytes.extend_from_slice(&sample.to_le_bytes());
    }

    Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header("X-Sample-Rate", sample_rate.to_string())
        .header("X-Sample-Format", "f32le")
        .header("X-Channels", "1")
        .body(pcm_bytes.into())
        .map_err(|e| SpeechError::Mp3Conversion(std::io::Error::other(e)))
}
