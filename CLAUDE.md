# Kokoros Directory (Text-to-Speech Service)

This directory contains the Kokoros text-to-speech (TTS) engine - a high-quality neural voice synthesis system that converts text into natural-sounding audio for the SpeakDoc application.

## Core Architecture

### Neural Text-to-Speech Engine
- **Technology Stack**: Rust-based TTS service with neural voice models
- **Voice Quality**: High-quality, natural-sounding speech synthesis
- **Multiple Voices**: Support for various voice personalities and styles
- **Real-time Processing**: Fast audio generation for responsive user experience
- **API Integration**: HTTP API server for TTS requests from SpeakDoc worker

### Service Components
- **kokoros**: Main TTS engine and service implementation
- **kokoros-openai**: OpenAI-compatible API interface for broader compatibility
- **koko**: Core voice processing library and utilities
- **checkpoints**: Neural model checkpoints and voice data
- **data**: Training data and voice model assets

## Component Structure

### Main TTS Engine (`kokoros/`)
- **Core TTS Service**: Primary text-to-speech processing engine
- **Voice Synthesis**: Neural network-based voice generation
- **Audio Processing**: High-quality audio encoding and output
- **Model Management**: Loading and management of voice model checkpoints

#### Key Features
- **Neural Voice Models**: Deep learning-based natural speech synthesis
- **Multiple Voice Support**: Various voice personalities and characteristics
- **High Audio Quality**: Professional-grade audio output
- **Batch Processing**: Efficient processing of multiple text segments
- **Streaming Output**: Real-time audio generation for large texts
- **SSML Support**: Industry-standard SSML tags for precise control:
  - `<break>` tags for configurable pauses and timing
  - `<emphasis>` tags for dynamic volume and speed adjustments

### OpenAI-Compatible API (`kokoros-openai/`)
- **API Compatibility**: OpenAI TTS API-compatible interface
- **Standard Integration**: Easy integration with existing TTS clients
- **Request Format**: Standard HTTP API with familiar request/response patterns
- **Voice Mapping**: Maps standard voice names to Kokoros voice models

#### API Endpoints
- **POST /v1/audio/speech**: Generate speech from text (OpenAI compatible)
- **POST /v1/audio/pcm**: Return raw f32le PCM audio with metadata headers (`X-Sample-Rate`, `X-Sample-Format`, `X-Channels`). Used by workers to avoid double-encoding — workers encode PCM to target formats using PyAV.
- **GET /health**: Service health check endpoint with detailed status information
- **GET /**: Simple health check returning "OK"

### Core Processing Library (`koko/`)
- **Audio Processing**: Core audio manipulation and encoding functions
- **Voice Processing**: Low-level voice synthesis algorithms
- **Model Utilities**: Helper functions for model loading and management
- **Format Support**: Multiple audio format encoding (WAV, MP3, etc.)

## Voice Models and Checkpoints

### Model Storage (`checkpoints/`)
- **Neural Model Files**: Pre-trained TTS model checkpoints
- **Voice Personalities**: Different voice characteristics and styles
- **Language Support**: Multi-language voice model support
- **Model Versioning**: Different versions of voice models for quality/performance trade-offs

### Voice Characteristics
- **Male/Female Voices**: Variety of gender-specific voice options
- **Accent Variations**: Different accent and regional voice styles
- **Speaking Styles**: Formal, casual, narrative, and expressive styles
- **Speed Variations**: Natural-sounding speech at different speeds

### Data Assets (`data/`)
- **Training Data**: Voice training datasets and phoneme mappings
- **Language Models**: Language-specific pronunciation and intonation data
- **Configuration Files**: Voice model configuration and parameter settings
- **Reference Audio**: Sample audio files for voice quality verification

## macOS Development

Auto-started by `start.speakdoc development run` as a host process.
To build: `start.speakdoc development build --kokoros` (runs `cargo build --release`).
To skip auto-start: `start.speakdoc development run --no-kokoros-autostart`.
To start manually: `cd Kokoros && cargo run --release --bin koko -- openai --port 3025`.
Logs to `kokoros.log` in Kokoros directory.
ONNX model loading takes ~120s on first start.

## Docker and Deployment

### Container Configuration (`Dockerfile`)
- **Rust-based Container**: Optimized container for Rust TTS service
- **Model Packaging**: Includes necessary voice models and data
- **GPU Support**: Optional GPU acceleration for faster processing
- **Port Configuration**: Exposed on port 3025 for SpeakDoc integration

### Installation and Setup
- `install.sh` - Installation script for dependencies and setup
- `download_all.sh` - Script to download all voice models and data
- **Dependency Management**: Automatic installation of required libraries
- **Model Download**: Automated download of voice model checkpoints

## Audio Processing Pipeline

### Text Processing
1. **Text Input**: Receive text from SpeakDoc worker service
2. **Pause Tag Processing**: Split text at `<pause>` and `<pause:N>` tags into segments
3. **Text Normalization**: Clean and normalize input text for TTS
4. **Phoneme Conversion**: Convert text to phonetic representation
5. **Prosody Analysis**: Determine stress, intonation, and rhythm patterns

### Voice Synthesis
1. **Model Loading**: Load appropriate voice model checkpoint
2. **Neural Processing**: Generate audio features using neural networks
3. **Vocoding**: Convert features to raw audio waveform
4. **Post-processing**: Apply filters and enhancement for audio quality

### Audio Output
1. **Format Encoding**: Encode audio in requested format (WAV, MP3)
2. **Quality Optimization**: Apply compression and quality settings
3. **Streaming Delivery**: Send audio back to requesting service
4. **Caching**: Optional caching of generated audio segments

## SSML Break Tag Support

Kokoros TTS supports SSML (Speech Synthesis Markup Language) `<break>` tags for configurable pauses at any position in text. This feature enables precise control over speech timing and pacing using industry-standard SSML syntax.

### SSML Break Tag Syntax

**Basic Break (default duration):**
```
"Hello there. <break/> How are you?"
```
- Uses default duration of 500ms (0.5 seconds)
- Inserts a natural pause between sentences or phrases

**Custom Break Duration with Time:**
```
"First part. <break time="2s"/> After a 2 second pause."
"Quick pause. <break time="500ms"/> Continue."
```
- Specify custom duration with `time` attribute
- Supports milliseconds (ms) or seconds (s) units
- Examples: `<break time="2s"/>` (2 seconds), `<break time="500ms"/>` (0.5 seconds)
- Decimal seconds supported: `<break time="2.5s"/>` (2.5 seconds)

**Break Duration with Strength:**
```
"Soft pause. <break strength="weak"/> Continue."
"Strong pause. <break strength="strong"/> Resume."
```
- Specify predefined strength levels
- Strength values map to specific durations (see table below)

**Multiple Breaks:**
```
"A <break time="250ms"/> B <break time="1s"/> C <break strength="strong"/> D"
```
- Supports unlimited break tags in a single text
- Each break can have its own custom time or strength

### Duration Guidelines

#### Time Attribute

Break duration can be specified with the `time` attribute using milliseconds (ms) or seconds (s):

| Tag | Duration | Use Case |
|-----|----------|----------|
| `<break time="100ms"/>` | 0.1 sec | Very brief pause between words |
| `<break time="250ms"/>` | 0.25 sec | Quick breath |
| `<break time="500ms"/>` | 0.5 sec | Natural sentence pause |
| `<break time="1s"/>` | 1.0 sec | Clear sentence break |
| `<break time="1.5s"/>` | 1.5 sec | Paragraph transition |
| `<break time="2s"/>` | 2.0 sec | Paragraph break, dramatic pause |
| `<break time="3s"/>` | 3.0 sec | Extended pause, section break |
| `<break time="5s"/>` | 5.0 sec | Very long pause |
| `<break/>` | 0.5 sec | **Default** - no attributes |

#### Strength Attribute

Predefined strength levels provide semantic pause durations:

| Tag | Duration | Use Case |
|-----|----------|----------|
| `<break strength="x-weak"/>` | 100ms | Minimal pause |
| `<break strength="weak"/>` | 250ms | Brief pause |
| `<break strength="medium"/>` | 500ms | **Default** - natural pause |
| `<break strength="strong"/>` | 1000ms | Emphatic pause |
| `<break strength="x-strong"/>` | 2000ms | Very emphatic pause |

### Implementation Details

**How It Works:**
1. **Tag Detection**: Regex pattern `<break(?:\s+(?:time="([^"]+)"|strength="([^"]+)"))*\s*/>` identifies SSML break tags in input text
2. **Attribute Parsing**: Extracts and parses `time` or `strength` attributes
3. **Duration Conversion**: Converts time units (s → ms) and strength values to millisecond durations
4. **Text Segmentation**: Text is split into segments at each break tag location
5. **Silent Audio Insertion**: Pure silence (zero samples) inserted at segment boundaries
6. **Audio Generation**: Each segment is processed separately and concatenated with silence
7. **Error Handling**: Invalid formats log errors and fall back to default 500ms duration

**Technical Implementation:**
- File: `kokoros/kokoros/src/tts/koko.rs`
- Function: `split_text_by_pauses()` - Parses SSML break tags and segments text
- Function: `time_to_duration_ms()` - Converts time attributes (ms/s) to milliseconds
- Function: `strength_to_duration_ms()` - Maps strength values to millisecond durations
- Function: `tts_raw_audio()` - Enhanced to insert silent audio between segments
- Default: `DEFAULT_BREAK_DURATION_MS = 500` (500ms)
- Method: Inserts silent audio samples (0.0 values) rather than using TTS tokens
- Formula: `silence_samples = sample_rate * duration_ms / 1000`

**Error Handling:**
- Invalid time values (e.g., `<break time="abc"/>`) → Log error, use default 500ms
- Invalid time units (e.g., `<break time="500h"/>`) → Log error, use default 500ms
- Invalid strength values (e.g., `<break strength="super"/>`) → Log error, use default 500ms
- Missing unit in time (e.g., `<break time="500"/>`) → Log error, use default 500ms

### API Usage

**OpenAI-Compatible Endpoint:**
```bash
curl -X POST http://localhost:3025/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello there. <break/> How are you? <break time=\"2s\"/> Let me continue.",
    "voice": "af_sky",
    "response_format": "wav"
  }' --output output.wav
```

**Request Parameters:**
- `input`: Text with embedded SSML `<break>` tags
- `voice`: Voice model to use (e.g., "af_sky")
- `speed`: Speech speed multiplier (default: 1.0)
- `initial_silence`: Optional silence duration at start in milliseconds (independent of break tags)
- `response_format`: Output format (wav, mp3, webm, aac)

### Use Cases

**Natural Pacing:**
```
"Welcome to our service. <break/> Let me explain how it works."
```
- Default 500ms pause for natural speech flow

**Dramatic Effect:**
```
"And the winner is... <break time=\"3s\"/> John Smith!"
```
- 3 second pause for dramatic tension

**List Reading:**
```
"Item one <break time=\"250ms\"/> Item two <break time=\"250ms\"/> Item three"
```
- Brief 250ms pauses between list items

**Section Breaks:**
```
"End of chapter one. <break time=\"2s\"/> Chapter two begins now."
```
- 2 second pause between major sections

**Emphasis with Strength:**
```
"This is <break strength=\"medium\"/> very <break strength=\"strong\"/> important."
```
- Semantic strength values for varying emphasis

**Mixed Time Units:**
```
"Quick. <break time=\"250ms\"/> Medium. <break time=\"1s\"/> Long. <break time=\"2.5s\"/> Done."
```
- Combine milliseconds and seconds as needed

### Worker Integration

The SpeakDoc worker service can include SSML break tags in text sent to Kokoros:

```python
# In worker text processing
processed_text = "Introduction to the topic. <break time=\"1s\"/> Now let's discuss details."
audio = kokoros_api.generate_speech(processed_text, voice="af_sky")
```

**Note**: Worker integration may require updates to generate SSML break tags instead of legacy pause tags.

### Testing

Unit tests verify SSML break tag functionality:

```bash
cd kokoros/kokoros
cargo test koko::tests
```

**Tests Included:**
- `test_break_tag_regex` - Validates SSML break tag pattern matching
- `test_time_to_duration_ms` - Validates time attribute parsing (ms, s, decimal s)
- `test_strength_to_duration_ms` - Validates strength attribute mapping
- `test_split_text_by_breaks_logic` - Validates text segmentation with SSML breaks

### Benefits

- ✅ **SSML Standard**: Uses industry-standard SSML syntax for broad compatibility
- ✅ **Precise Control**: Exact millisecond-level control over pause duration
- ✅ **Flexible Units**: Support for milliseconds (ms) and seconds (s), including decimals
- ✅ **Semantic Strength**: Predefined strength levels (x-weak to x-strong) for intuitive control
- ✅ **Easy Integration**: Standard SSML tags, no proprietary syntax
- ✅ **Error Resilience**: Invalid formats log errors and gracefully fall back to defaults
- ✅ **Clean Audio**: Inserts pure silence, no TTS artifacts or strange sounds
- ✅ **Natural Speech**: Improves pacing and comprehension

### Measuring Break Duration

An analysis tool is provided to measure actual pause durations in generated audio:

```bash
cd /path/to/kokoros
python3 analyze_pause.py output.wav
```

This tool detects silence regions and reports their timing, useful for verifying break behavior and calibrating duration values.

## SSML Emphasis Tag Support

Kokoros TTS supports SSML `<emphasis>` tags for controlling the prominence and intensity of specific words or phrases. This feature enables dynamic volume and speed adjustments to make important content stand out or de-emphasize less critical information.

### SSML Emphasis Tag Syntax

**Basic Emphasis (default level):**
```xml
"This is <emphasis>important</emphasis>."
```
- Uses default `moderate` level if no level attribute specified
- Makes text noticeably louder and slightly slower

**Emphasis with Specific Level:**
```xml
"This is <emphasis level=\"strong\">very important</emphasis>!"
"<emphasis level=\"reduced\">Side note:</emphasis> Continue with main content."
```
- Specify exact emphasis level with `level` attribute
- Five predefined levels available (see table below)

**Multiple Emphasis Tags:**
```xml
"We need <emphasis level=\"moderate\">better</emphasis> solutions, not <emphasis level=\"strong\">perfect</emphasis> ones."
```
- Supports unlimited emphasis tags in a single text
- Each tag can have its own level
- Works correctly with multiple tags on the same line

**Combined with Break Tags:**
```xml
"Listen carefully<break time=\"1s\"/> <emphasis level=\"x-strong\">This is critical!</emphasis>"
```
- Emphasis and break tags work seamlessly together
- Break provides pause, emphasis provides volume/speed control

### Emphasis Levels

Emphasis levels control both volume (amplitude) and speaking speed to create natural-sounding emphasis:

| Level | Tag Syntax | Volume | Speed | Description |
|-------|-----------|--------|-------|-------------|
| **none** | `<emphasis level="none">text</emphasis>` | 1.0x (normal) | 1.0x (normal) | Explicitly set to normal speech |
| **reduced** | `<emphasis level="reduced">text</emphasis>` | 0.5x (50% quieter) | 1.1x (faster) | De-emphasized, whisper-like, for side notes |
| **moderate** | `<emphasis level="moderate">text</emphasis>` | 1.3x (30% louder) | 1.0x (normal speed) | **DEFAULT** - Subtle emphasis, volume only |
| **strong** | `<emphasis level="strong">text</emphasis>` | 2.0x (2× louder) | 0.85x (slower) | Strong emphasis for important content |
| **x-strong** | `<emphasis level="x-strong">text</emphasis>` | 2.5x (2.5× louder) | 0.78x (much slower) | Maximum emphasis, very dramatic |

### Usage Examples

**Question Emphasis:**
```xml
<!-- Normal question -->
"How are you?"

<!-- Emphasized question -->
"<emphasis level=\"strong\">How are you?</emphasis>"

<!-- Urgent question -->
"<emphasis level=\"x-strong\">What have you done?</emphasis>"
```

**Content Hierarchy:**
```xml
"<emphasis level=\"reduced\">Note:</emphasis> The main point is <emphasis level=\"strong\">critical</emphasis>."
```
- `reduced` for parenthetical/side information
- `strong` for key points

**Dramatic Speech:**
```xml
"And the winner is<break time=\"2s\"/> <emphasis level=\"x-strong\">John Smith!</emphasis>"
```
- Combines pause for suspense with maximum emphasis for impact

**List with Varying Emphasis:**
```xml
"<emphasis level=\"moderate\">First</emphasis>, do this. <emphasis level=\"strong\">Second</emphasis>, do that. <emphasis level=\"x-strong\">Third</emphasis>, most important!"
```

### Implementation Details

**How It Works:**
1. **Tag Preprocessing**: SSML `<emphasis>` tags converted to internal markers
2. **Text Segmentation**: Text split into separate segments for each emphasis level
3. **Parallel Processing**: Each segment synthesized independently with adjusted parameters
4. **Volume Adjustment**: Audio amplitude multiplied by emphasis volume factor
5. **Speed Adjustment**: TTS synthesis speed modified by emphasis speed factor
6. **Audio Concatenation**: Segments merged into final audio output

**Technical Implementation:**
- File: `kokoros/kokoros/src/tts/koko.rs`
- Regex Pattern: `<emphasis(?:\s+level="([^"]+)")?\s*>(.*?)</emphasis>`
- Function: `preprocess_emphasis_tags()` - Converts tags to markers
- Function: `split_by_emphasis_markers()` - Handles multiple emphasis tags correctly
- Function: `emphasis_to_audio_params()` - Maps levels to volume/speed multipliers
- Audio Processing: Amplitude multiplication (volume) + speed parameter adjustment

**Error Handling:**
- Invalid emphasis levels (e.g., `<emphasis level="invalid">`) → Log error, use `moderate` as default
- Malformed tags → Ignored, text processed normally
- Volume clipping → Audio samples clamped to prevent distortion

### API Usage

**OpenAI-Compatible Endpoint:**
```bash
curl -X POST http://localhost:3025/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "This is <emphasis level=\"strong\">very important</emphasis>!",
    "voice": "af_sky",
    "response_format": "wav"
  }' --output output.wav
```

**Request Parameters:**
- `input`: Text with embedded SSML `<emphasis>` tags
- `voice`: Voice model to use
- `speed`: Base speech speed multiplier (emphasis adjustments apply on top of this)
- `response_format`: Output format (wav, mp3, webm, aac)

### Worker Integration

The SpeakDoc worker service can include SSML emphasis tags in text sent to Kokoros:

```python
# In worker text processing
processed_text = "Introduction. <emphasis level=\"strong\">Key point here.</emphasis> Continue."
audio = kokoros_api.generate_speech(processed_text, voice="af_sky")
```

### Testing

Unit tests verify SSML emphasis tag functionality:

```bash
cd kokoros/kokoros
cargo test koko::tests
```

**Tests Included:**
- `test_emphasis_tag_regex` - Validates SSML emphasis tag pattern matching
- `test_parse_emphasis_level` - Validates emphasis level string parsing
- `test_emphasis_to_audio_params` - Validates volume/speed parameter mapping
- `test_extract_emphasis_from_text` - Validates emphasis marker extraction

### Use Cases

- **Questions**: Add urgency or emphasis to questions
- **Warnings**: Make critical warnings stand out
- **Hierarchical Content**: De-emphasize side notes, emphasize main points
- **Dramatic Effect**: Create suspense or impact with x-strong emphasis
- **Accessibility**: Help listeners identify important content
- **Natural Pacing**: Vary speech dynamics for more engaging audio

### Benefits

- ✅ **SSML Standard**: Uses industry-standard SSML syntax for broad compatibility
- ✅ **Dynamic Volume**: Precise control over audio amplitude (0.5x to 2.5x)
- ✅ **Speed Variation**: Coordinated speed changes for natural emphasis
- ✅ **Multiple Tags**: Correctly handles unlimited emphasis tags per text
- ✅ **Easy Integration**: Standard SSML tags, no proprietary syntax
- ✅ **Combines with Breaks**: Works seamlessly with `<break>` tags
- ✅ **Natural Sound**: Volume + speed adjustments mimic human emphasis patterns

## API Integration with SpeakDoc

### Worker Service Integration
1. **TTS Request**: Worker sends text and voice parameters to Kokoros. SpeakDoc always sends `speed=1.0`; playback speed is controlled browser-side via `playbackRate`. SSML `<emphasis>` tags still vary speed per-segment for natural emphasis.
2. **Voice Selection**: Specify desired voice model and characteristics
3. **Audio Generation**: Kokoros processes text and generates audio
4. **Response Delivery**: High-quality audio returned to worker

### Request Format
```json
{
  "text": "Text to synthesize",
  "voice": "voice_name",
  "speed": 1.0,
  "format": "wav",
  "quality": "high"
}
```

### Response Format
- **Audio Data**: Raw audio bytes or streaming audio
- **Metadata**: Audio duration, format, and quality information
- **Error Handling**: Detailed error messages for processing failures

### Health Check Endpoint

The `/health` endpoint provides detailed service status information:

**Response Format:**
```json
{
  "status": "healthy",
  "service": "kokoros-tts",
  "model_loaded": true,
  "sample_rate": 24000,
  "supported_formats": ["wav", "mp3"],
  "pool_size": 4
}
```

**Response Fields:**
- `status`: Current service health status ("healthy" when operational)
- `service`: Service identifier ("kokoros-tts")
- `model_loaded`: Boolean indicating if TTS model is loaded and ready
- `sample_rate`: Audio sample rate in Hz (typically 24000)
- `supported_formats`: Array of supported audio output formats
- `pool_size`: Number of ONNX Runtime sessions in the pool for parallel processing

## Voice Configuration

### Available Voices
- **Professional Voices**: Clear, formal speaking voices for business content
- **Narrative Voices**: Engaging voices optimized for storytelling and articles
- **Conversational Voices**: Natural, casual voices for dialogue and interaction
- **Specialized Voices**: Technical, academic, or domain-specific voice styles

### Voice Parameters
- **Speed Control**: Adjustable speaking rate (0.5x to 2.0x normal speed)
- **Pitch Adjustment**: Voice pitch modification for tone variation
- **Volume Control**: Audio level and dynamic range adjustment
- **Emphasis Control**: Stress and emphasis pattern customization

## Performance Optimization

### Rust Performance Benefits
- **Memory Efficiency**: Low memory footprint for large-scale processing
- **CPU Optimization**: Efficient CPU usage for real-time synthesis
- **Concurrent Processing**: Multi-threaded processing for batch requests
- **Zero-Copy Operations**: Efficient audio data handling

### ONNX Runtime Thread Configuration

**CRITICAL**: When running multiple Kokoros instances, proper thread pool configuration is essential to prevent CPU contention and timeout errors.

#### Thread Pool Architecture
ONNX Runtime uses three independent threading systems:
- **OpenMP threads**: Controls CPU kernel parallelization
- **ONNX Runtime intra-op threads**: Parallelizes operations within neural network nodes
- **ONNX Runtime inter-op threads**: Runs different neural network nodes in parallel

#### Configuration Requirements
Multi-instance deployment requires both:
1. **Environment variables** set in docker-compose configuration
2. **Programmatic configuration** in `kokoros/src/onn/ort_base.rs` SessionBuilder

**Critical Note**: Environment variables alone are insufficient. ONNX Runtime's Rust bindings require programmatic API calls to enforce thread limits. Without code-level configuration, all instances will compete for all available CPU cores regardless of environment variable settings.

#### Impact of Proper Configuration

**Without thread limiting**:
- CPU overload: instances use 10-11 cores each instead of intended 8 cores
- System capacity exceeded (e.g., 40+ cores used on 32-core system)
- Audio generation timeouts (70-110+ seconds, exceeding 120s nginx timeout)
- 504 Gateway errors requiring automatic retries
- ~5-6% request failure rate on first attempt

**With proper thread limiting**:
- Each instance uses ~8 cores as intended
- Total CPU stays within system limits (~31 cores on 32-core system)
- Audio generation: 20-55 seconds (~3x performance improvement)
- Zero timeout errors
- 100% success rate without retries

### Multi-Instance Deployment

Kokoros supports horizontal scaling with multiple instances behind nginx load balancing:

#### Deployment Architecture
- **Instance Count**: Supports 1-4 instances (recommendation for 32-core system)
- **Thread Allocation**: Total CPU cores divided evenly among instances (e.g., 32 cores ÷ 4 = 8 cores per instance)
- **Load Balancing**: Nginx round-robin distribution across all healthy instances
- **Session Pooling**: Each instance maintains a pool of ONNX Runtime sessions (default: 4) for internal parallelization
- **Health Monitoring**: `/health` endpoint reports instance status and configuration

#### Scaling Guidelines
- **Single Instance**: Development/testing, 100% CPU utilization possible
- **Dual Instance**: Balanced production, moderate load distribution
- **Triple Instance**: High-performance production
- **Quad Instance**: Maximum throughput, optimal for 32-core systems

#### Performance Characteristics
Processing speed improves with additional instances (up to system core count):
- Single instance: ~2 minutes per page
- Dual instance: ~1.5 minutes per page
- Triple instance: ~1.2 minutes per page
- Quad instance: ~1 minute per page

### Caching and Optimization
- **Model Caching**: Keep frequently used models in memory
- **Audio Caching**: Cache generated audio segments for repeated text
- **Batch Processing**: Process multiple text segments efficiently
- **Streaming Processing**: Generate audio in chunks for large texts

## Quality Control

### Audio Quality Metrics
- **Naturalness**: Human-like speech patterns and intonation
- **Intelligibility**: Clear pronunciation and articulation
- **Consistency**: Stable voice characteristics throughout synthesis
- **Emotional Expression**: Appropriate emotional tone and expression

### Testing and Validation
- **Regression Testing**: Automated testing of voice quality
- **A/B Testing**: Comparison of different voice models and settings
- **User Feedback**: Integration with SpeakDoc feedback system
- **Performance Monitoring**: Audio generation speed and resource usage

## Logging and Monitoring

### Service Monitoring
- **Request Logging**: TTS request tracking and analytics
- **Performance Metrics**: Audio generation speed and latency
- **Error Tracking**: Synthesis failure logging and analysis
- **Resource Monitoring**: CPU, memory, and GPU usage tracking

### Quality Assurance
- **Audio Validation**: Automatic quality checks on generated audio
- **Model Health**: Neural model performance monitoring
- **Service Health**: API endpoint availability and response times
- **Error Rate Tracking**: Synthesis success/failure rate monitoring

## Development and Scripts

### Development Tools (`scripts/`)
- **Model Training**: Scripts for training custom voice models
- **Audio Processing**: Utilities for audio format conversion and analysis
- **Testing Tools**: Voice quality testing and validation scripts
- **Deployment**: Production deployment and configuration scripts

### Build System
- `Cargo.toml` - Rust workspace configuration and dependencies
- `Cargo.lock` - Dependency version lock for reproducible builds
- Component-specific build configurations for each service
- Cross-compilation support for different deployment targets

## Integration Notes

### SpeakDoc Integration
- **Port Configuration**: Runs on port 3025 for worker service communication
- **API Compatibility**: RESTful HTTP API for easy integration
- **Error Handling**: Graceful error handling and retry mechanisms
- **Performance**: Optimized for SpeakDoc's audio generation requirements

### External Compatibility
- **OpenAI API**: Compatible with standard OpenAI TTS API clients
- **Audio Formats**: Support for WAV, MP3, and other common formats
- **Voice Standards**: Compatible with standard voice naming conventions
- **Streaming Support**: Real-time audio streaming for large content

## File Organization

```
kokoros/
├── kokoros/                 # Main TTS engine
│   ├── src/                # Core TTS implementation
│   └── Cargo.toml         # Main engine dependencies
├── kokoros-openai/          # OpenAI-compatible API
│   ├── src/                # API server implementation
│   └── Cargo.toml         # API server dependencies
├── koko/                    # Core processing library
│   ├── src/                # Audio processing utilities
│   └── Cargo.toml         # Library dependencies
├── checkpoints/             # Neural model checkpoints
├── data/                    # Training data and assets
├── scripts/                 # Development and deployment scripts
├── target/                  # Rust build output
├── Cargo.toml              # Workspace configuration
├── Dockerfile              # Container build configuration
├── install.sh              # Installation script
├── download_all.sh         # Model download script
├── README.md               # Project documentation
└── CLAUDE.md               # This comprehensive documentation
```

## Troubleshooting

### 504 Gateway Timeout Errors

**Symptoms**:
- Worker receives HTML 504 error responses instead of audio
- Audio generation exceeding 120-second nginx timeout
- CPU usage per instance significantly exceeds configured limit
- Multiple automatic retry attempts for same audio segments

**Common Root Cause**: ONNX Runtime thread pools not properly configured

**Verification Steps**:
1. Check environment variables are set in docker-compose configuration
2. Verify programmatic thread configuration exists in `kokoros/src/onn/ort_base.rs`
3. Confirm Kokoros logs show thread configuration on startup
4. Monitor CPU usage distribution across instances (should be roughly equal)

**Expected Behavior**:
- Each instance using allocated CPU cores (e.g., ~800% for 8 cores)
- Total system CPU within capacity (e.g., ~3100% for 4 instances on 32-core system)
- Audio generation completing in 20-60 seconds
- No timeout errors or automatic retries

### Performance Monitoring

**Key Metrics**:
- **Processing Time**: Audio generation should complete in 20-60 seconds per segment
- **CPU Distribution**: Instances should show balanced CPU usage
- **Error Rate**: No audio error logs should be created for successful processing
- **Success Rate**: 100% completion without retries indicates optimal configuration

### Health Check Verification

Health endpoint (`/health`) provides service status including:
- Service operational status
- Model load state
- Audio sample rate and format support
- ONNX Runtime session pool size

## Development Notes

- **Model Requirements**: Voice models must be downloaded before first use
- **GPU Acceleration**: Optional GPU support for faster processing
- **Memory Usage**: Voice models require significant memory for optimal performance
- **Audio Quality**: Balance between quality and processing speed based on use case
- **Integration**: Critical component for SpeakDoc's text-to-speech functionality
- **Port Configuration**: Default port 3025 for Docker deployment
- **Thread Configuration**: MUST configure ONNX Runtime threads programmatically for multi-instance deployment