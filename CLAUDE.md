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

### OpenAI-Compatible API (`kokoros-openai/`)
- **API Compatibility**: OpenAI TTS API-compatible interface
- **Standard Integration**: Easy integration with existing TTS clients
- **Request Format**: Standard HTTP API with familiar request/response patterns
- **Voice Mapping**: Maps standard voice names to Kokoros voice models

#### API Endpoints
- **POST /v1/audio/speech**: Generate speech from text (OpenAI compatible)
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
2. **Text Normalization**: Clean and normalize input text for TTS
3. **Phoneme Conversion**: Convert text to phonetic representation
4. **Prosody Analysis**: Determine stress, intonation, and rhythm patterns

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

## API Integration with SpeakDoc

### Worker Service Integration
1. **TTS Request**: Worker sends text and voice parameters to Kokoros
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
  "supported_formats": ["wav", "mp3"]
}
```

**Response Fields:**
- `status`: Current service health status ("healthy" when operational)
- `service`: Service identifier ("kokoros-tts")
- `model_loaded`: Boolean indicating if TTS model is loaded and ready
- `sample_rate`: Audio sample rate in Hz (typically 24000)
- `supported_formats`: Array of supported audio output formats

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

## Development Notes

- **Model Requirements**: Voice models must be downloaded before first use
- **GPU Acceleration**: Optional GPU support for faster processing
- **Memory Usage**: Voice models require significant memory for optimal performance
- **Audio Quality**: Balance between quality and processing speed based on use case
- **Integration**: Critical component for SpeakDoc's text-to-speech functionality
- **Port Configuration**: Default port 3025 for Docker deployment