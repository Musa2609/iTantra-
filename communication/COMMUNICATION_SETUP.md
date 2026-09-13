# iTantra Communication Prototype

## Setup

Use Python 3.12 for the ASR/TTS pipeline:

```powershell
py -3.12 -m venv voice-env
& ".\voice-env\Scripts\python.exe" -m pip install -r requirements-voice.txt
```

The AI4Bharat ASR and TTS models require Hugging Face access and are downloaded on first use.

## ASR, translation, ggwave, and TTS

Hindi speech to English speech:

```powershell
& ".\voice-env\Scripts\python.exe" -u test_asr_tts_ggwave.py `
  --audio audio.wav `
  --language hi `
  --source-language hin_Deva `
  --target-language eng_Latn `
  --tts-output received_tts_english.wav
```

## Two-laptop acoustic test

Run the receiver first:

```powershell
& ".\voice-env\Scripts\python.exe" -u test_asr_tts_ggwave.py `
  --physical-receive `
  --target-language eng_Latn `
  --tts-output received_physical_tts.wav
```

Then run the sender:

```powershell
& ".\voice-env\Scripts\python.exe" -u test_asr_tts_ggwave.py `
  --physical-send `
  --audio audio.wav `
  --language hi `
  --source-language hin_Deva `
  --target-language eng_Latn
```

The sender performs ASR and translation, then plays ggwave tones. The receiver decodes the text and synthesizes speech in the target language.

## Acoustic packet transport software loopback

```powershell
& ".\voice-env\Scripts\python.exe" -u test_ggwave.py
```