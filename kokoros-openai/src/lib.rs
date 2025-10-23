use std::error::Error;
use std::io::{self};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, routing::get, routing::post};
use kokoros::{
    tts::koko::{InitConfig as TTSKokoInitConfig, TTSKoko},
    utils::mp3::pcm_to_mp3,
    utils::wav::{WavHeader, write_audio_chunk},
};
use log::{debug, info};
use serde::{Deserialize, Serialize};
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

/// Session pool for parallel ONNX Runtime processing
#[derive(Clone)]
struct SessionPool {
    sessions: Arc<Vec<TTSKoko>>,
    counter: Arc<AtomicUsize>,
}

impl SessionPool {
    fn new(sessions: Vec<TTSKoko>) -> Self {
        let pool_size = sessions.len();
        info!(
            "Created session pool with {} ONNX Runtime sessions",
            pool_size
        );
        Self {
            sessions: Arc::new(sessions),
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get next session using round-robin selection
    fn get_session(&self) -> &TTSKoko {
        let index = self.counter.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        &self.sessions[index]
    }

    fn pool_size(&self) -> usize {
        self.sessions.len()
    }
}

/// Shared server state with session pooling and concurrency limiting
#[derive(Clone)]
struct ServerState {
    session_pool: SessionPool,
    /// Semaphore to limit concurrent TTS requests
    /// Should match or exceed session pool size for true parallelism
    concurrency_limit: Arc<Semaphore>,
}

impl ServerState {
    fn new(session_pool: SessionPool, max_concurrent_requests: usize) -> Self {
        let pool_size = session_pool.pool_size();

        info!(
            "Initializing TTS server with {} ONNX sessions and max {} concurrent requests",
            pool_size, max_concurrent_requests
        );

        if max_concurrent_requests < pool_size {
            info!(
                "WARNING: max_concurrent ({}) < pool_size ({}). Consider increasing KOKOROS_MAX_CONCURRENT to {}",
                max_concurrent_requests, pool_size, pool_size
            );
        }

        Self {
            session_pool,
            concurrency_limit: Arc::new(Semaphore::new(max_concurrent_requests)),
        }
    }
}

pub async fn create_server(sessions: Vec<TTSKoko>) -> Router {
    debug!("create_server()");

    let pool_size = sessions.len();

    // Default max_concurrent to pool_size for true parallelism
    let max_concurrent = std::env::var("KOKOROS_MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(pool_size);

    let session_pool = SessionPool::new(sessions);
    let state = ServerState::new(session_pool, max_concurrent);

    Router::new()
        .route("/", get(handle_home))
        .route("/health", get(handle_health))
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

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    service: String,
    model_loaded: bool,
    sample_rate: u32,
    supported_formats: Vec<String>,
    pool_size: usize,
}

/// Returns detailed health information about the TTS service
async fn handle_health(State(state): State<ServerState>) -> Json<HealthResponse> {
    let config = TTSKokoInitConfig::default();

    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "kokoros-tts".to_string(),
        model_loaded: true,
        sample_rate: config.sample_rate,
        supported_formats: vec!["wav".to_string(), "mp3".to_string()],
        pool_size: state.session_pool.pool_size(),
    })
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

    // Get next available session from pool (round-robin)
    let session = state.session_pool.get_session();

    let raw_audio = session
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
