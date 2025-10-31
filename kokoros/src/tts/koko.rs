use crate::onn::ort_koko::{self};
use crate::tts::tokenize::tokenize;
use crate::utils;
use log::{debug, error, info};
use ndarray::Array3;
use ndarray_npy::NpzReader;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use espeak_rs::text_to_phonemes;

/// Represents a text segment with an optional pause duration (in milliseconds) to prepend
#[derive(Debug, Clone)]
struct TextSegment {
    text: String,
    pause_tokens: Option<usize>, // Actually pause duration in milliseconds
}

/// Default pause duration when <pause> tag is used without explicit duration
/// Value is in milliseconds: 500ms = 0.5 seconds
const DEFAULT_PAUSE_TOKENS: usize = 500;

#[derive(Debug, Clone)]
pub struct TTSOpts<'a> {
    pub txt: &'a str,
    pub lan: &'a str,
    pub style_name: &'a str,
    pub save_path: &'a str,
    pub mono: bool,
    pub speed: f32,
    pub initial_silence: Option<usize>,
}

#[derive(Clone)]
pub struct TTSKoko {
    #[allow(dead_code)]
    model_path: String,
    model: Arc<Mutex<ort_koko::OrtKoko>>,
    styles: HashMap<String, Vec<[[f32; 256]; 1]>>,
    init_config: InitConfig,
}

#[derive(Clone)]
pub struct InitConfig {
    pub model_url: String,
    pub voices_url: String,
    pub sample_rate: u32,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            model_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx".into(),
            voices_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin".into(),
            sample_rate: 24000,
        }
    }
}

impl TTSKoko {
    pub async fn new(model_path: &str, voices_path: &str) -> Self {
        Self::from_config(model_path, voices_path, InitConfig::default()).await
    }

    pub async fn from_config(model_path: &str, voices_path: &str, cfg: InitConfig) -> Self {
        if !Path::new(model_path).exists() {
            utils::fileio::download_file_from_url(cfg.model_url.as_str(), model_path)
                .await
                .expect("download model failed.");
        }

        if !Path::new(voices_path).exists() {
            utils::fileio::download_file_from_url(cfg.voices_url.as_str(), voices_path)
                .await
                .expect("download voices data file failed.");
        }

        let model = Arc::new(Mutex::new(
            ort_koko::OrtKoko::new(model_path.to_string())
                .expect("Failed to create Kokoro TTS model"),
        ));
        // TODO: if(not streaming) { model.print_info(); }
        // model.print_info();

        let styles = Self::load_voices(voices_path);

        TTSKoko {
            model_path: model_path.to_string(),
            model,
            styles,
            init_config: cfg,
        }
    }

    /// Splits text by <pause> and <pause:N> tags into segments with pause information
    ///
    /// The pause tag applies to the text segment that comes AFTER it.
    /// The N value represents milliseconds of pause.
    ///
    /// Examples:
    /// - "Hello <pause> world" -> [("Hello", None), ("world", Some(500))]  // 500ms default
    /// - "A <pause:1000> B <pause:2000> C" -> [("A", None), ("B", Some(1000)), ("C", Some(2000))]  // 1s and 2s
    fn split_text_by_pauses(&self, text: &str) -> Vec<TextSegment> {
        // Regex to match <pause> or <pause:N> tags
        let pause_regex = Regex::new(r"<pause(?::(\d+))?>").unwrap();

        let mut segments = Vec::new();
        let mut last_end = 0;
        let mut pending_pause: Option<usize> = None;

        for cap in pause_regex.captures_iter(text) {
            let match_obj = cap.get(0).unwrap();
            let match_start = match_obj.start();
            let match_end = match_obj.end();

            // Extract the text before this pause tag
            let text_segment = &text[last_end..match_start];

            // Add the text segment with any pending pause from the previous tag
            if !text_segment.trim().is_empty() {
                segments.push(TextSegment {
                    text: text_segment.to_string(),
                    pause_tokens: pending_pause,
                });
            }

            // Parse pause duration from capture group (or use default)
            // This pause will apply to the NEXT segment
            pending_pause = Some(
                cap.get(1)
                    .and_then(|m| m.as_str().parse::<usize>().ok())
                    .unwrap_or(DEFAULT_PAUSE_TOKENS),
            );

            last_end = match_end;
        }

        // Add remaining text after last pause tag, with any pending pause
        if last_end < text.len() {
            let remaining_text = &text[last_end..];
            if !remaining_text.trim().is_empty() {
                segments.push(TextSegment {
                    text: remaining_text.to_string(),
                    pause_tokens: pending_pause,
                });
            }
        }

        // If no pause tags were found, return the entire text as a single segment
        if segments.is_empty() && !text.trim().is_empty() {
            segments.push(TextSegment {
                text: text.to_string(),
                pause_tokens: None,
            });
        }

        debug!("Split text into {} segments with pauses", segments.len());
        for (i, seg) in segments.iter().enumerate() {
            debug!(
                "  Segment {}: pause_tokens={:?}, text_len={}",
                i,
                seg.pause_tokens,
                seg.text.len()
            );
        }

        segments
    }

    fn split_text_into_chunks(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut chunks = Vec::new();

        // First split by sentences - using common sentence ending punctuation
        let sentences: Vec<&str> = text
            .split(['.', '?', '!', ';'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();

        for sentence in sentences {
            // Clean up the sentence and add back punctuation
            let sentence = format!("{}.", sentence.trim());

            // Convert to phonemes to check token count
            let sentence_phonemes = text_to_phonemes(&sentence, "en", None, true, false)
                .unwrap_or_default()
                .join("");
            let token_count = tokenize(&sentence_phonemes).len();

            if token_count > max_tokens {
                // If single sentence is too long, split by words
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    let test_chunk = if word_chunk.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", word_chunk, word)
                    };

                    let test_phonemes = text_to_phonemes(&test_chunk, "en", None, true, false)
                        .unwrap_or_default()
                        .join("");
                    let test_tokens = tokenize(&test_phonemes).len();

                    if test_tokens > max_tokens {
                        if !word_chunk.is_empty() {
                            chunks.push(word_chunk);
                        }
                        word_chunk = word.to_string();
                    } else {
                        word_chunk = test_chunk;
                    }
                }

                if !word_chunk.is_empty() {
                    chunks.push(word_chunk);
                }
            } else if !current_chunk.is_empty() {
                // Try to append to current chunk
                let test_text = format!("{} {}", current_chunk, sentence);
                let test_phonemes = text_to_phonemes(&test_text, "en", None, true, false)
                    .unwrap_or_default()
                    .join("");
                let test_tokens = tokenize(&test_phonemes).len();

                if test_tokens > max_tokens {
                    // If combining would exceed limit, start new chunk
                    chunks.push(current_chunk);
                    current_chunk = sentence;
                } else {
                    current_chunk = test_text;
                }
            } else {
                current_chunk = sentence;
            }
        }

        // Add the last chunk if not empty
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    pub fn tts_raw_audio(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // First, split text by pause tags
        let segments = self.split_text_by_pauses(txt);
        let mut final_audio = Vec::new();

        for (segment_idx, segment) in segments.iter().enumerate() {
            debug!(
                "Processing segment {}/{}: pause_tokens={:?}, text_len={}",
                segment_idx + 1,
                segments.len(),
                segment.pause_tokens,
                segment.text.len()
            );

            // Determine the silence duration for this segment
            // First segment uses the passed-in initial_silence
            // Subsequent segments use pause_tokens from the pause tag
            let silence_tokens = if segment_idx == 0 {
                initial_silence
            } else {
                segment.pause_tokens
            };

            // If this segment needs silence before it, insert silent audio samples
            if let Some(pause_duration_ms) = silence_tokens {
                // The pause tag value directly represents milliseconds
                // e.g., <pause:1000> = 1 second, <pause:500> = 0.5 seconds
                let silence_duration_ms = pause_duration_ms as f32;
                let silence_samples =
                    (self.init_config.sample_rate as f32 * silence_duration_ms / 1000.0) as usize;

                // Insert silent audio (zeros)
                let silent_audio = vec![0.0_f32; silence_samples];
                final_audio.extend_from_slice(&silent_audio);

                debug!(
                    "Inserted {}ms of silence ({} samples)",
                    silence_duration_ms, silence_samples
                );
            }

            // Split this segment's text into appropriate chunks for processing
            let max_chunk_tokens = 500 - 20; // Extra 20 token safety margin (no longer need space for silence tokens)
            let chunks = self.split_text_into_chunks(&segment.text, max_chunk_tokens.max(100)); // Minimum 100 tokens

            for (_chunk_idx, chunk) in chunks.iter().enumerate() {
                // Convert chunk to phonemes
                let phonemes = text_to_phonemes(chunk, lan, None, true, false)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?
                    .join("");
                debug!("phonemes: {}", phonemes);
                let tokens = tokenize(&phonemes);

                // Get style vectors once
                let styles = self.mix_styles(style_name, tokens.len())?;

                // pad a 0 to start and end of tokens
                let mut padded_tokens = vec![0];
                for &token in &tokens {
                    padded_tokens.push(token);
                }
                padded_tokens.push(0);

                let tokens = vec![padded_tokens];

                match self
                    .model
                    .lock()
                    .unwrap()
                    .infer(tokens, styles.clone(), speed)
                {
                    Ok(chunk_audio) => {
                        let chunk_audio: Vec<f32> = chunk_audio.iter().cloned().collect();
                        final_audio.extend_from_slice(&chunk_audio);
                    }
                    Err(e) => {
                        error!("Error processing chunk: {:?}", e);
                        error!("Chunk text was: {:?}", chunk);
                        return Err(Box::new(std::io::Error::other(format!(
                            "Chunk processing failed: {:?}",
                            e
                        ))));
                    }
                }
            }
        }

        Ok(final_audio)
    }

    pub fn tts(
        &self,
        TTSOpts {
            txt,
            lan,
            style_name,
            save_path,
            mono,
            speed,
            initial_silence,
        }: TTSOpts,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let audio = self.tts_raw_audio(txt, lan, style_name, speed, initial_silence)?;

        // Save to file
        if mono {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        } else {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        info!("Audio saved to {}", save_path);
        Ok(())
    }

    pub fn mix_styles(
        &self,
        style_name: &str,
        tokens_len: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !style_name.contains("+") {
            if let Some(style) = self.styles.get(style_name) {
                let styles = vec![style[tokens_len][0].to_vec()];
                Ok(styles)
            } else {
                Err(format!("can not found from styles_map: {}", style_name).into())
            }
        } else {
            debug!("parsing style mix");
            let styles: Vec<&str> = style_name.split('+').collect();

            let mut style_names = Vec::new();
            let mut style_portions = Vec::new();

            for style in styles {
                // Note: Using nested if-let instead of let chains for stable Rust compatibility
                #[allow(clippy::collapsible_if)]
                if let Some((name, portion)) = style.split_once('.') {
                    if let Ok(portion) = portion.parse::<f32>() {
                        style_names.push(name);
                        style_portions.push(portion * 0.1);
                    }
                }
            }
            debug!("styles: {:?}, portions: {:?}", style_names, style_portions);

            let mut blended_style = vec![vec![0.0; 256]; 1];

            for (name, portion) in style_names.iter().zip(style_portions.iter()) {
                if let Some(style) = self.styles.get(*name) {
                    let style_slice = &style[tokens_len][0]; // This is a [256] array
                    // Blend into the blended_style
                    for (j, &value) in style_slice.iter().enumerate().take(256) {
                        blended_style[0][j] += value * portion;
                    }
                }
            }
            Ok(blended_style)
        }
    }

    fn load_voices(voices_path: &str) -> HashMap<String, Vec<[[f32; 256]; 1]>> {
        let mut npz = NpzReader::new(File::open(voices_path).unwrap()).unwrap();
        let mut map = HashMap::new();

        for voice in npz.names().unwrap() {
            let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
            let voice_data = voice_data.unwrap();
            let mut tensor = vec![[[0.0; 256]; 1]; 511];
            for (i, inner_value) in voice_data.outer_iter().enumerate() {
                for (j, inner_inner_value) in inner_value.outer_iter().enumerate() {
                    for (k, number) in inner_inner_value.iter().enumerate() {
                        tensor[i][j][k] = *number;
                    }
                }
            }
            map.insert(voice, tensor);
        }

        let sorted_voices = {
            let mut voices = map.keys().collect::<Vec<_>>();
            voices.sort();
            voices
        };

        info!("voice styles loaded: {:?}", sorted_voices);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_tag_regex() {
        // Test basic pause tag detection
        let pause_regex = Regex::new(r"<pause(?::(\d+))?>").unwrap();

        // Test <pause> without duration
        let text = "Hello <pause> world";
        let matches: Vec<_> = pause_regex.captures_iter(text).collect();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].get(1).is_none()); // No duration specified

        // Test <pause:30> with duration
        let text = "Hello <pause:30> world";
        let matches: Vec<_> = pause_regex.captures_iter(text).collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].get(1).unwrap().as_str(), "30");

        // Test multiple pauses
        let text = "A <pause> B <pause:10> C <pause:50> D";
        let matches: Vec<_> = pause_regex.captures_iter(text).collect();
        assert_eq!(matches.len(), 3);
        assert!(matches[0].get(1).is_none());
        assert_eq!(matches[1].get(1).unwrap().as_str(), "10");
        assert_eq!(matches[2].get(1).unwrap().as_str(), "50");
    }

    #[test]
    fn test_split_text_by_pauses_logic() {
        // Note: This test doesn't require a full TTSKoko instance
        // We're just testing the regex logic directly

        let pause_regex = Regex::new(r"<pause(?::(\d+))?>").unwrap();

        // Test simple case
        let text = "Hello <pause> world";
        let mut segments = Vec::new();
        let mut last_end = 0;

        for cap in pause_regex.captures_iter(text) {
            let match_obj = cap.get(0).unwrap();
            let before = &text[last_end..match_obj.start()];
            segments.push(before);
            last_end = match_obj.end();
        }
        segments.push(&text[last_end..]);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], "Hello ");
        assert_eq!(segments[1], " world");
    }
}
