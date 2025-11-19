# VieNeu-TTS

Vietnamese Text-to-Speech (TTS) model with voice cloning capabilities.

## Table of Contents
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install espeak-ng](#2-install-espeak-ng)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
- [Voice Cloning](#voice-cloning)
- [Usage](#usage)
  - [Running the Test Script](#running-the-test-script)
  - [Using CPU Instead of CUDA](#using-cpu-instead-of-cuda)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd VieNeu-TTS
```

### 2. Install espeak-ng

espeak-ng is required for text phonemization. Install it based on your operating system:

#### Windows

Download and install the latest Windows installer from the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases).

After installation, add espeak-ng to your system PATH:
1. Find the installation directory (usually `C:\Program Files\eSpeak NG\`)
2. Add this directory to your system PATH environment variable
3. Verify installation by running: `espeak-ng --version`

Alternatively, using Chocolatey:
```bash
choco install espeak-ng
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install espeak-ng
```

#### macOS

Using Homebrew:
```bash
brew install espeak-ng
```

#### Verify Installation

After installation, verify that espeak-ng is properly installed:
```bash
espeak-ng --version
```

### 3. Install Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Required packages:**
- `phonemizer>=3.3.0` - Text to phoneme conversion
- `torch` - Deep learning framework
- `torchvision` - Vision utilities for PyTorch
- `torchaudio` - Audio processing for PyTorch
- `gradio` - Web interface framework
- `neucodec>=0.0.4` - Neural audio codec
- `librosa>=0.11.0` - Audio processing library

## Voice Cloning

To clone a voice, you need to provide:
1. A reference audio file (WAV format, preferably 16kHz)
2. The corresponding transcript of the reference audio

### Prepare Your Reference Files

Create a directory (e.g., `sample/`) and add:
- `your_voice.wav` - A clean audio recording (10-30 seconds recommended)
- `your_voice.txt` - The exact transcript of what was said in the audio

**Example:**
```
sample/
├── duyanh.wav
├── duyanh.txt
├── vanbac.wav
└── vanbac.txt
```

The text file should contain the exact Vietnamese text spoken in the audio file.

## Usage

### Running the Test Script

The repository includes a test script (`test_tts.py`) that demonstrates basic usage:

```bash
python test_tts.py
```

This will:
1. Load the reference audio and text from `sample/duyanh.wav` and `sample/duyanh.txt`
2. Initialize the VieNeu-TTS model
3. Generate speech for a sample text
4. Save the output to `output.wav`

### Using CPU Instead of CUDA

If you don't have CUDA support (NVIDIA GPU), you need to modify the device settings to use CPU.

#### Option 1: Edit the test_tts.py file

Open `test_tts.py` and change the device parameters:

```python
# Initialize TTS model
tts = VietnameseTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cpu",  # Changed from "cuda" to "cpu"
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"  # Changed from "cuda" to "cpu"
)
```

#### Option 2: Create your own script

```python
from pathlib import Path
from vietnamese_tts import VietnameseTTS
from utils.normalize_text import VietnameseTTSNormalizer
import soundfile as sf

# Initialize normalizer
normalizer = VietnameseTTSNormalizer()

# Initialize TTS model with CPU
tts = VietnameseTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"
)

# Reference audio and text
ref_audio = "sample/duyanh.wav"
ref_text = Path("sample/duyanh.txt").read_text(encoding="utf-8")
ref_text_norm = normalizer.normalize(ref_text)

# Encode reference audio
ref_codes = tts.encode_reference(ref_audio)

# Text to synthesize
text = "Công nghệ giọng nói đang phát triển rất nhanh."
text_norm = normalizer.normalize(text)

# Generate speech
wav = tts.infer(text_norm, ref_codes, ref_text_norm)

# Save output
sf.write("output.wav", wav, 24000)
print("Speech generated successfully!")
```

## API Reference

### VietnameseTTS

Main TTS class for speech synthesis.

**Initialization:**
```python
tts = VietnameseTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",  # or "pnnbao-ump/VieNeu-TTS-1000h"
    backbone_device="cuda",  # or "cpu"
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"  # or "cpu"
)
```

**Methods:**

- `encode_reference(ref_audio_path: str)` - Encode reference audio for voice cloning
  - Returns: Reference codes for the audio

- `infer(text: str, ref_codes: np.ndarray, ref_text: str)` - Generate speech
  - `text`: Normalized text to synthesize
  - `ref_codes`: Encoded reference audio
  - `ref_text`: Normalized reference text
  - Returns: Audio waveform as numpy array (24kHz)

### VietnameseTTSNormalizer

Text normalization for Vietnamese.

```python
from utils.normalize_text import VietnameseTTSNormalizer

normalizer = VietnameseTTSNormalizer()
normalized_text = normalizer.normalize("Xin chào!")
```

## Troubleshooting

### espeak-ng not found

**Error:** `Command 'espeak-ng' not found` or similar phonemization errors.

**Solution:** Make sure espeak-ng is installed and available in your system PATH. See the [Install espeak-ng](#2-install-espeak-ng) section.

### CUDA out of memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Switch to CPU mode or use a machine with more GPU memory. See [Using CPU Instead of CUDA](#using-cpu-instead-of-cuda).

### Slow inference on CPU

**Note:** Inference on CPU is significantly slower than on GPU. For a typical sentence, expect:
- GPU: ~2-5 seconds
- CPU: ~30-60 seconds or more

Consider using a smaller model or running on a GPU-enabled machine for better performance.

### Module import errors

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Audio quality issues

For best results:
- Use clean reference audio without background noise
- Reference audio should be 10-30 seconds long
- Ensure the transcript exactly matches the spoken text
- Use a sampling rate of 16kHz for reference audio

## License

Please refer to the model repository for licensing information.

## Acknowledgments

- Based on the VieNeu-TTS model by pnnbao-ump
- Uses NeuCodec for audio encoding/decoding
