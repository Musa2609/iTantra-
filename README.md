# iTantra — Acoustic Speech Communication Bridge

End-to-end acoustic pipeline that lets **two laptops exchange speech** using only sound waves (speakers + microphone) — no internet, no Wi-Fi, no Bluetooth. Works offline after the one-time model download.

```
Laptop 1 (Sender)
  Your voice
    -> AI4Bharat IndicConformer ASR      (speech -> text in your Indian language)
    -> AI4Bharat IndicTrans2             (optional: translate to another language)
    -> CRC32 packet + ggwave tones       (text -> acoustic data tones over speaker)

Laptop 2 (Receiver) -- listening through microphone
    -> ggwave demodulate                 (acoustic tones -> recovered text + CRC check)
    -> AI4Bharat Indic Parler-TTS        (text -> synthesised speech WAV)
  Playback of received speech
```

Supported languages: **Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese** (plus Sanskrit, Nepali, Dogri, etc.)

---

## Requirements

- **Python 3.12** (exact version -- IndicConformer ONNX binaries require it)
- A **Hugging Face account** with access to the four AI4Bharat models listed below
- Speakers + microphone on each laptop
- Windows, Linux, or macOS

---

## One-Time Setup (Do This On Every Laptop)

### 1. Clone the repository

```bash
git clone https://github.com/abdulakibsiddiqui/communication
cd communication
```

### 2. Create the Python 3.12 virtual environment

**Windows (PowerShell):**
```powershell
py -3.12 -m venv voice-env
.\voice-env\Scripts\python.exe -m pip install --upgrade pip
.\voice-env\Scripts\python.exe -m pip install -r requirements-voice.txt
```

**Linux / macOS:**
```bash
python3.12 -m venv voice-env
./voice-env/bin/python -m pip install --upgrade pip
./voice-env/bin/python -m pip install -r requirements-voice.txt
```

### 3. Request access to the AI4Bharat models on Hugging Face

Visit each page and click **"Request access"** (approved instantly):

| Model | Purpose |
|---|---|
| ai4bharat/indic-conformer-600m-multilingual | ASR + Language Detection |
| ai4bharat/indic-parler-tts | Text-to-Speech synthesis |
| ai4bharat/indictrans2-indic-en-dist-200M | Translation (Indic -> English) |
| ai4bharat/indictrans2-en-indic-dist-200M | Translation (English -> Indic) |

### 4. Log in to Hugging Face

```powershell
.\voice-env\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli login
```

Paste your Hugging Face token when prompted. Saved locally, only needed once.

> Models are downloaded on first run (~2.5 GB total) and cached at ~/.cache/huggingface/

---

## SENDER MODE

Place your speakers close to the receiver laptop microphone, then run:

```powershell
# Speak any Indian language -- auto-detected, no translation
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-send --record 5 --auto-detect

# Speak any language -> translate to English before sending
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-send --record 5 --auto-detect --target-language eng_Latn

# Speak any language -> translate to Hindi before sending
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-send --record 5 --auto-detect --target-language hin_Deva

# Speak any language -> translate to Gujarati before sending
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-send --record 5 --auto-detect --target-language guj_Gujr
```

---

## RECEIVER MODE

**Start the receiver BEFORE the sender plays tones.** The receiver listens on the microphone.

```powershell
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-receive --tts-output received_speech.wav
```

The receiver will:
1. Listen via microphone for incoming ggwave acoustic tones
2. Validate the CRC32 packet integrity
3. Print the recovered text
4. Generate received_speech.wav (spoken audio of the message)
5. Print: PHYSICAL ASR/TTS TEST: PASS

---

## Language Reference

### Auto-detect supported languages

| Language | IndicTrans2 Target Tag |
|---|---|
| Hindi | hin_Deva |
| Bengali | ben_Beng |
| Tamil | tam_Taml |
| Telugu | tel_Telu |
| Marathi | mar_Deva |
| Gujarati | guj_Gujr |
| Kannada | kan_Knda |
| Malayalam | mal_Mlym |
| Punjabi | pan_Guru |
| Odia | ory_Orya |
| Assamese | asm_Beng |
| English (target only) | eng_Latn |

---

## Software Loopback Test (single laptop)

```powershell
.\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --record 5 --auto-detect --target-language eng_Latn --tts-output received_tts_english.wav
```

---

## How It Works

| Component | Model |
|---|---|
| Language Detection + ASR | ai4bharat/indic-conformer-600m-multilingual (ONNX, offline) |
| Translation | ai4bharat/indictrans2-* (greedy decode, fast CPU) |
| Acoustic Encoding | ggwave FSK tones |
| TTS | ai4bharat/indic-parler-tts |

Text is wrapped in a CRC32 packet before encoding: [ITANTRA_PKT_V1][packet_number][payload][CRC32].
If tones are corrupted the CRC catches it and raises an error instead of delivering wrong text.

> NOTE: Speak in Hindi, Tamil, Bengali, Marathi, Gujarati, etc. AI4Bharat models do not support English speech input.
