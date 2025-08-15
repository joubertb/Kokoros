# Kokoros GitHub Actions Configuration

## Docker Hub Setup

To enable automatic container builds and pushes to Docker Hub, configure the following secrets in your GitHub repository settings:

### Required Secrets

1. **DOCKERHUB_USERNAME**
   - Your Docker Hub username
   - Example: `joubertb`

2. **DOCKERHUB_TOKEN** 
   - Docker Hub access token (not your password)
   - Create at: https://hub.docker.com/settings/security
   - Use "Read, Write, Delete" permissions

### Setting up secrets:

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with the exact name and value

## Workflow Details

### Kokoros Container Workflow
- **File**: `.github/workflows/build-kokoros.yml`
- **Triggers**: 
  - Push to `pdftotts` branch (excludes documentation changes)
  - Pull requests to `pdftotts` branch
- **Features**:
  - Multi-platform builds (linux/amd64, linux/arm64)
  - Docker layer caching via GitHub Actions cache
  - Registry cache for faster subsequent builds
  - Service-specific tagging with `kokoros-` prefix

### Image Tags

The workflow creates the following tags:
- `kokoros-latest` - Latest build from pdftotts branch
- `kokoros-pdftotts-<git-sha>` - Specific commit builds from pdftotts branch
- `kokoros-pr-<number>` - Pull request builds (not pushed to registry)

## Integration with SpeakDoc

This Kokoros TTS service is designed to integrate with the SpeakDoc application:

### Docker Hub Repository
- **Repository**: `joubertb/repos`
- **Image Tags**: `kokoros-latest`, `kokoros-main-<sha>`
- **Usage**: Pull image for SpeakDoc deployment

### SpeakDoc Integration
The main SpeakDoc application references this image in its `docker-compose-speakdoc.yml`:
```yaml
kokoros:
  image: joubertb/repos:kokoros-latest
```

## Build Process

### Rust Build Pipeline
1. **Dependency Installation**: Install Rust toolchain and system dependencies
2. **Model Download**: Download neural TTS models and voice data
3. **Compilation**: Build optimized Rust binaries for TTS service
4. **Container Assembly**: Package service with models into runtime container

### Build Optimization
- **Multi-stage Build**: Separate build and runtime environments
- **Caching**: GitHub Actions and registry caching for faster builds
- **Cross-platform**: Builds for both AMD64 and ARM64 architectures
- **Size Optimization**: Minimal runtime container with only necessary components

## Service Configuration

### Port Configuration
- **Default Port**: 3025
- **API Endpoint**: OpenAI-compatible TTS API
- **Health Check**: `/health` endpoint for service monitoring

### Performance Notes
- **Model Loading**: Voice models loaded at startup (may take time)
- **Memory Requirements**: Significant memory needed for neural models
- **CPU/GPU**: Optimized for CPU inference, GPU acceleration optional

## Development Notes

- **Local Development**: Use `cargo build --release` for local compilation
- **Model Requirements**: Voice models must be downloaded before building
- **Container Size**: Large container due to neural TTS models
- **Build Time**: Extended build time due to Rust compilation and model download