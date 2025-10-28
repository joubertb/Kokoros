use log::{debug, info};
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::CoreMLExecutionProvider;
use ort::execution_providers::cpu::CPUExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use std::env;

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let _providers = [CUDAExecutionProvider::default().build()];

        #[cfg(feature = "coreml")]
        let _providers = [
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ];

        #[cfg(all(not(feature = "cuda"), not(feature = "coreml")))]
        let _providers = [CPUExecutionProvider::default().build()];

        // Read thread configuration from environment variables
        let intra_threads = env::var("ORT_INTRA_OP_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0); // 0 means use default
        let inter_threads = env::var("ORT_INTER_OP_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0); // 0 means use default

        info!(
            "Configuring ONNX Runtime with intra_threads={}, inter_threads={}",
            intra_threads, inter_threads
        );

        match SessionBuilder::new() {
            Ok(mut builder) => {
                // Configure thread pools if environment variables are set
                if intra_threads > 0 {
                    builder = builder
                        .with_intra_threads(intra_threads)
                        .map_err(|e| format!("Failed to set intra threads: {}", e))?;
                }
                if inter_threads > 0 {
                    builder = builder
                        .with_inter_threads(inter_threads)
                        .map_err(|e| format!("Failed to set inter threads: {}", e))?;
                }

                let session = builder
                    .with_execution_providers(_providers)
                    .map_err(|e| format!("Failed to build session: {}", e))?
                    .commit_from_file(model_path)
                    .map_err(|e| format!("Failed to commit from file: {}", e))?;
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
            info!("Configured with: CUDA execution provider");

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
