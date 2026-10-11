use log::{debug, info, warn};
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::{
    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat, CoreMLSpecializationStrategy,
};
use ort::execution_providers::cpu::CPUExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

/// Check if NVIDIA CUDA is available by looking for the driver library.
/// This prevents a native crash when CUDA EP tries to initialize without drivers.
#[cfg(feature = "cuda")]
fn is_cuda_available() -> bool {
    use std::path::Path;

    // Check for nvidia-smi (most reliable indicator of working drivers)
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        if output.success() {
            return true;
        }
    }

    // Fallback: check for libcuda.so in standard paths
    let cuda_lib_paths = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/lib/libcuda.so",
    ];
    for path in &cuda_lib_paths {
        if Path::new(path).exists() {
            return true;
        }
    }

    false
}

/// Resolve the per-model CoreML cache directory.
///
/// Default root: `<OS cache dir>/kokoros/coreml/` (e.g. `~/Library/Caches/kokoros/coreml/` on macOS).
/// Override or disable via `KOKOROS_COREML_CACHE_DIR` env var ("none"/"disable" → no cache).
///
/// The model-specific subdirectory is keyed by `<file-size>-<mtime-secs>` so that replacing the
/// ONNX file with a new version automatically triggers a fresh CoreML compilation.
#[cfg(feature = "coreml")]
fn coreml_cache_dir(model_path: &str) -> Option<std::path::PathBuf> {
    let root: Option<std::path::PathBuf> =
        match std::env::var("KOKOROS_COREML_CACHE_DIR").ok().as_deref() {
            Some("none") | Some("disable") | Some("") => return None,
            Some(path) => Some(std::path::PathBuf::from(path)),
            None => dirs::cache_dir().map(|d| d.join("kokoros").join("coreml")),
        };

    let root = root?;

    let meta = std::fs::metadata(model_path).ok()?;
    let size = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let key = format!("{}-{}", size, mtime);
    let dir = root.join(key);

    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(e) => {
            warn!(
                "CoreML cache dir {:?} could not be created ({}); continuing without cache",
                dir, e
            );
            None
        }
    }
}

/// Parses a thread-count environment value. Unset, empty, zero and non-numeric
/// values yield `None`, which leaves the ONNX Runtime default in place.
fn parse_thread_count(value: Option<&str>) -> Option<usize> {
    value?.trim().parse::<usize>().ok().filter(|n| *n > 0)
}

fn thread_count_from_env(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok();
    let count = parse_thread_count(value.as_deref());
    if count.is_none() {
        if let Some(raw) = value.filter(|v| !v.trim().is_empty()) {
            warn!("Ignoring invalid {} '{}', using ONNX Runtime default", name, raw);
        }
    }
    count
}

/// Applies `ORT_INTRA_OP_NUM_THREADS` / `ORT_INTER_OP_NUM_THREADS` to a session.
///
/// With no explicit size ONNX Runtime creates one thread per host core and pins
/// each to that core. Inside a container limited by cpuset the pool is oversized
/// (and every `ModelPool` slot gets its own) and pinning to cores outside the set
/// fails (pthread_setaffinity_np: Invalid argument). An explicit size disables
/// the pinning.
fn with_thread_counts(mut builder: SessionBuilder) -> Result<SessionBuilder, String> {
    if let Some(threads) = thread_count_from_env("ORT_INTRA_OP_NUM_THREADS") {
        info!("ONNX Runtime intra-op threads: {}", threads);
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| format!("Failed to set intra-op threads: {}", e))?;
    }
    if let Some(threads) = thread_count_from_env("ORT_INTER_OP_NUM_THREADS") {
        info!("ONNX Runtime inter-op threads: {}", threads);
        builder = builder
            .with_inter_threads(threads)
            .map_err(|e| format!("Failed to set inter-op threads: {}", e))?;
    }
    Ok(builder)
}

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        {
            if is_cuda_available() {
                info!("NVIDIA CUDA drivers detected, attempting GPU acceleration");
                let cuda_providers = [
                    CUDAExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ];
                match SessionBuilder::new() {
                    Ok(builder) => {
                        match with_thread_counts(builder)?
                            .with_execution_providers(cuda_providers)
                            .map_err(|e| format!("Failed to build session: {}", e))?
                            .commit_from_file(&model_path)
                        {
                            Ok(session) => {
                                info!("Model loaded with CUDA execution provider");
                                self.set_sess(session);
                                return Ok(());
                            }
                            Err(e) => {
                                warn!("CUDA model loading failed, falling back to CPU: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Session builder failed with CUDA, falling back to CPU: {}",
                            e
                        );
                    }
                }
            } else {
                info!("No NVIDIA CUDA drivers detected, using CPU execution provider");
            }
        }

        #[cfg(feature = "coreml")]
        let _providers = {
            let cache_dir = coreml_cache_dir(&model_path);
            let mlprogram = std::env::var("KOKOROS_COREML_MLPROGRAM")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            let profile = std::env::var("KOKOROS_COREML_PROFILE_COMPUTE_PLAN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            // KOKOROS_COREML_COMPUTE_UNITS: all (default) | gpu | ane
            let compute_units = match std::env::var("KOKOROS_COREML_COMPUTE_UNITS")
                .as_deref()
                .unwrap_or("all")
            {
                "gpu" => Some(CoreMLComputeUnits::CPUAndGPU),
                "ane" => Some(CoreMLComputeUnits::CPUAndNeuralEngine),
                _ => None, // all compute units (CoreML default)
            };

            let fast_prediction = std::env::var("KOKOROS_COREML_FAST_PREDICTION")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            let mut provider = CoreMLExecutionProvider::default();
            if let Some(dir) = &cache_dir {
                provider = provider.with_model_cache_dir(dir.display().to_string());
                info!("CoreML model cache: {}", dir.display());
            }
            if mlprogram {
                provider = provider.with_model_format(CoreMLModelFormat::MLProgram);
                info!("CoreML model format: MLProgram");
            }
            if let Some(units) = compute_units {
                provider = provider.with_compute_units(units);
            }
            if fast_prediction {
                provider = provider
                    .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction);
            }
            if profile {
                provider = provider.with_profile_compute_plan(true);
            }
            vec![provider.build(), CPUExecutionProvider::default().build()]
        };

        #[cfg(not(feature = "coreml"))]
        let _providers = vec![CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(builder) => {
                let session = with_thread_counts(builder)?
                    .with_execution_providers(_providers)
                    .map_err(|e| format!("Failed to build session: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
                #[cfg(feature = "coreml")]
                info!("Model loaded with CoreML execution provider (CPU fallback)");
                #[cfg(not(feature = "coreml"))]
                info!("Model loaded with CPU execution provider");
                self.set_sess(session);
                Ok(())
            }
            Err(e) => Err(format!("Failed to create session builder: {}", e)),
        }
    }

    fn print_info(&self) {
        if let Some(session) = self.sess() {
            debug!("Input names:");
            for input in &session.inputs {
                debug!("  - {}", input.name);
            }
            debug!("Output names:");
            for output in &session.outputs {
                debug!("  - {}", output.name);
            }

            #[cfg(feature = "cuda")]
            info!("Configured with: CUDA execution provider (CPU fallback enabled)");

            #[cfg(feature = "coreml")]
            info!("Configured with: CoreML execution provider");

            #[cfg(all(not(feature = "cuda"), not(feature = "coreml")))]
            info!("Configured with: CPU execution provider");
        } else {
            debug!("Session is not initialized.");
        }
    }

    fn set_sess(&mut self, sess: Session);
    fn sess(&self) -> Option<&Session>;
}

#[cfg(test)]
mod tests {
    use super::parse_thread_count;

    #[test]
    fn parses_positive_counts() {
        assert_eq!(parse_thread_count(Some("4")), Some(4));
        assert_eq!(parse_thread_count(Some(" 2 ")), Some(2));
    }

    #[test]
    fn rejects_unset_zero_and_invalid_values() {
        assert_eq!(parse_thread_count(None), None);
        assert_eq!(parse_thread_count(Some("")), None);
        assert_eq!(parse_thread_count(Some("0")), None);
        assert_eq!(parse_thread_count(Some("-1")), None);
        assert_eq!(parse_thread_count(Some("four")), None);
    }
}
