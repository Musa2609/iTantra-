import sys
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import AutoModel

# Force UTF-8 output encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Load AI4Bharat model
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
)

# Load audio using soundfile
data, sr = sf.read("audio.wav", dtype="float32")

# Convert to mono if multi-channel
if data.ndim > 1:
    data = data.mean(axis=1)

# Convert to torch tensor of shape (1, samples)
wav = torch.from_numpy(data).unsqueeze(0)

# Resample to 16 kHz if needed
target_sample_rate = 16000
if sr != target_sample_rate:
    resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
    wav = resampler(wav)

# Hindi = "hi"
transcription = model(wav, "hi", "ctc")

print("Transcription:")
print(transcription)