use log::{debug, info, warn};
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::CoreMLExecutionProvider;
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
                        match builder
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
        let _providers = [
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ];

        #[cfg(not(feature = "coreml"))]
        let _providers = [CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(builder) => {
                let session = builder
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
