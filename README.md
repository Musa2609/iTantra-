# iTantra — Prototype for SIH

iTantra is an intelligent audio communication pipeline prototype developed for Smart India Hackathon (SIH).

## Project Setup & Modules

### 1. AI4Bharat IndicConformer Speech-to-Text (ASR)
- **Model**: `ai4bharat/indic-conformer-600m-multilingual`
- **File**: `test_asr.py`
- **Audio Processing**: Converts `audio.wav` (mono, float32, 16 kHz) using `soundfile` without relying on C++/FFmpeg binaries.
- **Output**: Direct Hindi transcription from audio.

#### Run ASR Test:
```bash
python -u test_asr.py
```

---

### 2. Meta EnCodec Neural Audio Compression
- **Model**: Pretrained Meta EnCodec (24 kHz model)
- **File**: `test_encodec.py`
- **Bitrates Tested**: 24.0 kbps, 12.0 kbps, 6.0 kbps, 3.0 kbps, 1.5 kbps
- **Functionality**:
  - Encodes raw audio into vector quantized (VQ) discrete codes.
  - Decodes VQ code frames back into playable audio.
  - Saves reconstructed audio as `reconstructed_<bitrate>kbps.wav`.
  - Achieves up to **940x compression ratio** vs raw PCM audio.

#### Run EnCodec Test:
```bash
python -u test_encodec.py
```

---

## Directory Contents
- `test_asr.py` — IndicConformer ASR test script
- `test_encodec.py` — Meta EnCodec compression test script
- `audio.wav` — Original test audio
- `reconstructed_*.wav` — Output reconstructed audio files at various bitrates
