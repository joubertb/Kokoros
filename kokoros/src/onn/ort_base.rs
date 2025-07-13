#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::coreml::CoreMLExecutionProvider;
use ort::execution_providers::cpu::CPUExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use log::{debug, info};

pub trait OrtBase {
    fn load_model(&mut self, model_path: String) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];

        #[cfg(feature = "coreml")]
        let providers = [
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build()
        ];

        #[cfg(all(not(feature = "cuda"), not(feature = "coreml")))]
        let providers = [CPUExecutionProvider::default().build()];

        match SessionBuilder::new() {
            Ok(builder) => {
                let session = builder
                    .with_execution_providers(providers)
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
