from pathlib import Path
from vietnamese_tts import VietnameseTTS
from utils.normalize_text import VietnameseTTSNormalizer
import soundfile as sf

# Reference audio and text
ref_audio = "sample/duyanh.wav"
ref_text = Path("sample/duyanh.txt").read_text(encoding="utf-8")

# Initialize normalizer
normalizer = VietnameseTTSNormalizer()
ref_text_norm = normalizer.normalize(ref_text)

print(f"Reference text: {ref_text}")
print(f"Normalized reference text: {ref_text_norm}")

# Initialize TTS model
print("\nInitializing VieNeu-TTS model...")
tts = VietnameseTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device="cuda",  # Change to "cpu" if you don't have CUDA
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"  # Change to "cpu" if you don't have CUDA
)

# Encode reference audio
print("Encoding reference audio...")
ref_codes = tts.encode_reference(ref_audio)

# Text to synthesize
text = "Công nghệ giọng nói đang phát triển rất nhanh."
text_norm = normalizer.normalize(text)

print(f"\nInput text: {text}")
print(f"Normalized text: {text_norm}")

# Generate speech
print("\nGenerating speech...")
wav = tts.infer(text_norm, ref_codes, ref_text_norm)

# Save output
output_file = "output.wav"
sf.write(output_file, wav, 24000)
print(f"\n✓ Speech generated successfully!")
print(f"✓ Output saved to: {output_file}")
