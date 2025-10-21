use std::error::Error;
use std::io::{self};
use std::sync::Arc;

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, routing::get, routing::post};
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::mp3::pcm_to_mp3,
    utils::wav::{WavHeader, write_audio_chunk},
};
use log::{debug, info};
use serde::Deserialize;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

#[derive(Deserialize, Default, Debug)]
#[serde(rename_all = "lowercase")]
enum AudioFormat {
    #[default]
    Wav,
    Mp3,
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

    input: String,

    #[serde(default)]
    voice: Voice,

    // Must be WAV
    #[allow(dead_code)]
    #[serde(default)]
    response_format: AudioFormat,

    #[serde(default)]
    speed: Speed,

    #[serde(default)]
    initial_silence: Option<usize>,
}

/// Shared server state with concurrency limiting
#[derive(Clone)]
struct ServerState {
    tts: TTSKoko,
    /// Semaphore to limit concurrent TTS requests
    /// Default: 4 concurrent requests max to prevent ONNX Runtime overload
    concurrency_limit: Arc<Semaphore>,
}

impl ServerState {
    fn new(tts: TTSKoko, max_concurrent_requests: usize) -> Self {
        info!(
            "Initializing TTS server with max {} concurrent requests",
            max_concurrent_requests
        );
        Self {
            tts,
            concurrency_limit: Arc::new(Semaphore::new(max_concurrent_requests)),
        }
    }
}

pub async fn create_server(tts: TTSKoko) -> Router {
    debug!("create_server()");

    // Default to 4 concurrent requests to prevent overwhelming ONNX Runtime
    // This can be made configurable via environment variable if needed
    let max_concurrent = std::env::var("KOKOROS_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let state = ServerState::new(tts, max_concurrent);

    Router::new()
        .route("/", get(handle_home))
        .route("/v1/audio/speech", post(handle_tts))
        .layer(CorsLayer::permissive())
        .with_state(state)
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
    State(state): State<ServerState>,
    Json(SpeechRequest {
        model: _,
        input,
        voice: Voice(voice),
        response_format,
        speed: Speed(speed),
        initial_silence,
    }): Json<SpeechRequest>,
) -> Result<Response, SpeechError> {
    // Acquire semaphore permit to limit concurrency
    // This will wait if max concurrent requests are already processing
    let _permit = state.concurrency_limit.acquire().await.map_err(|e| {
        SpeechError::Koko(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to acquire concurrency permit: {}", e),
        )))
    })?;

    debug!(
        "Processing TTS request for voice: {}, speed: {}, text length: {}",
        voice,
        speed,
        input.len()
    );

    let raw_audio = state
        .tts
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
                pcm_to_mp3(&raw_audio, sample_rate).map_err(|e| SpeechError::Mp3Conversion(e))?;

            ("audio/mpeg", mp3_data)
        }
    };

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .body(audio_data.into())
        .map_err(|e| {
            SpeechError::Mp3Conversion(std::io::Error::new(std::io::ErrorKind::Other, e))
        })?)
}
