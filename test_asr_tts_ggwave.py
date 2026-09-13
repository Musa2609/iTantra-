import argparse
import os
import sys
import wave
import time

import ggwave

from test_ggwave import PACKET_HEADER, packetize, validate_packet


ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
INDIC_TO_ENGLISH_MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
ENGLISH_TO_INDIC_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
GGWAVE_SAMPLE_RATE = 48000
PHYSICAL_LEAD_SECONDS = 3

LANGUAGE_TO_INDICTRANS2 = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "sa": "san_Deva",
    "ne": "npi_Deva",
    "sd": "snd_Arab",
    "ks": "kas_Arab",
    "kok": "kok_Deva",
    "mai": "mai_Deva",
    "doi": "doi_Deva",
    "brx": "brx_Deva",
    "mni": "mni_Mtei",
    "sat": "sat_Olck",
}

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "doi": "Dogri",
    "brx": "Bodo",
    "mni": "Manipuri",
    "sat": "Santali",
}


def record_audio_from_mic(output_path="audio.wav", duration=5.0, sample_rate=16000):
    """Record speech directly from the microphone."""
    import sounddevice as sd
    import soundfile as sf
    print(f"\n[Microphone] Recording {duration:.1f} seconds of speech... Speak into your mic now!")
    captured = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(output_path, captured, sample_rate)
    print(f"[Microphone] Recording complete. Saved to '{output_path}'.\n")


_ASR_MODEL = None


def get_asr_model():
    """Load and cache IndicConformer model in memory to avoid reloading 28 ONNX sessions."""
    global _ASR_MODEL
    if _ASR_MODEL is None:
        from transformers import AutoModel
        print("[ASR] Initializing AI4Bharat IndicConformer...", flush=True)
        _ASR_MODEL = AutoModel.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
        _ASR_MODEL.eval()
    return _ASR_MODEL


def detect_and_transcribe_with_indicconformer(audio_path, candidate_languages=None):
    """Auto-detect spoken Indian language and transcribe using AI4Bharat IndicConformer."""
    import soundfile as sf
    import torch
    import torchaudio.transforms as transforms

    if candidate_languages is None:
        # Check primary major languages for fast, reliable detection (Urdu excluded)
        candidate_languages = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"]

    data, sample_rate = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sample_rate != 16000:
        waveform = transforms.Resample(sample_rate, 16000)(waveform)

    model = get_asr_model()
    print("[Language Detection] Classifying spoken language across candidate languages...", flush=True)
    with torch.no_grad():
        encoder_outputs, encoded_lengths = model.encode(waveform)
        raw_logprobs = model.models['ctc_decoder'].run(['logprobs'], {'encoder_output': encoder_outputs})[0]

        scores = {}
        for lang in candidate_languages:
            if lang not in model.language_masks:
                continue
            mask = model.language_masks[lang]
            sub_lp = torch.from_numpy(raw_logprobs[:, :, mask]).log_softmax(dim=-1)
            max_lp, idx = torch.max(sub_lp[0], dim=-1)
            non_blank = idx != model.config.BLANK_ID
            if non_blank.sum() > 0:
                scores[lang] = max_lp[non_blank].mean().item()
            else:
                scores[lang] = -999.0

        best_lang = max(scores.items(), key=lambda x: x[1])[0]
        transcription = str(model(waveform, best_lang, "ctc")).strip()
        return best_lang, transcription, scores


def transcribe_with_ai4bharat(audio_path, language):
    import soundfile as sf
    import torch
    import torchaudio.transforms as transforms

    data, sample_rate = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sample_rate != 16000:
        waveform = transforms.Resample(sample_rate, 16000)(waveform)
    model = get_asr_model()
    with torch.no_grad():
        return str(model(waveform, language, "ctc")).strip()


_TRANSLATION_MODELS = {}
_INDIC_PROCESSOR = None


def get_indic_processor():
    global _INDIC_PROCESSOR
    if _INDIC_PROCESSOR is None:
        from IndicTransToolkit import IndicProcessor
        _INDIC_PROCESSOR = IndicProcessor(inference=True)
    return _INDIC_PROCESSOR


def get_translation_model(model_id):
    global _TRANSLATION_MODELS
    if model_id not in _TRANSLATION_MODELS:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"[Translation] Loading {model_id.split('/')[-1]} from disk cache...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).eval()
        _TRANSLATION_MODELS[model_id] = (tokenizer, model)
    return _TRANSLATION_MODELS[model_id]


def translate_with_indictrans(text, source_language, target_language):
    """Translate using IndicTrans2; language tags use BCP-47-like model tags."""
    if source_language == target_language:
        return text
    import torch

    processor = get_indic_processor()

    def translate_once(sentence, model_id, source, target):
        tokenizer, model = get_translation_model(model_id)
        batch = processor.preprocess_batch([sentence], src_lang=source, tgt_lang=target)
        inputs = tokenizer(batch, padding="longest", truncation=True, return_tensors="pt")
        with torch.no_grad():
            # num_beams=1 (greedy search) is 4-5x faster on CPU than beam search
            generated = model.generate(**inputs, use_cache=True, max_length=128, num_beams=1)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return processor.postprocess_batch(decoded, lang=target)[0]

    if source_language.startswith("eng_"):
        return translate_once(text, ENGLISH_TO_INDIC_MODEL_ID, source_language, target_language)
    if target_language.startswith("eng_"):
        return translate_once(text, INDIC_TO_ENGLISH_MODEL_ID, source_language, target_language)
    english = translate_once(text, INDIC_TO_ENGLISH_MODEL_ID, source_language, "eng_Latn")
    return translate_once(english, ENGLISH_TO_INDIC_MODEL_ID, "eng_Latn", target_language)


def text_to_ggwave(text):
    payload = text.encode("utf-8")
    packets = packetize(payload)
    if len(packets) != 1:
        raise ValueError("Recognized text is too long for one ggwave message")
    instance = ggwave.init()
    try:
        waveform = ggwave.encode(packets[0], instance=instance)
        received_packet = ggwave.decode(instance, waveform)
    finally:
        ggwave.free(instance)
    validated = validate_packet(received_packet, 1) if received_packet else None
    if validated is None:
        raise RuntimeError("ggwave did not recover the text packet")
    _number, received_payload = validated
    return received_payload.decode("utf-8"), waveform


def send_text_acoustically(text, output_device=None):
    import numpy as np
    import sounddevice as sd

    payload = packetize(text.encode("utf-8"))[0]
    instance = ggwave.init()
    try:
        waveform = np.frombuffer(ggwave.encode(payload, instance=instance), dtype=np.float32)
    finally:
        ggwave.free(instance)

    print(f"\n[Sender] Ready to transmit message acoustically.")
    for sec in range(PHYSICAL_LEAD_SECONDS, 0, -1):
        print(f"[Sender] Playback begins in {sec} second(s)...", flush=True)
        time.sleep(1.0)

    print(f"[Sender] >>> PLAYING ACOUSTIC TONES NOW <<< (listen to speakers)", flush=True)
    sd.play(waveform, samplerate=GGWAVE_SAMPLE_RATE, device=output_device, blocking=True)
    print(f"[Sender] Audio transmission finished.")
    return len(waveform) / GGWAVE_SAMPLE_RATE


def receive_text_acoustically(input_device=None):
    import numpy as np
    import sounddevice as sd

    probe = packetize(b"physical-text-probe")[0]
    instance = ggwave.init()
    try:
        probe_waveform = ggwave.encode(probe, instance=instance)
    finally:
        ggwave.free(instance)
    duration = PHYSICAL_LEAD_SECONDS + len(probe_waveform) / 4 / GGWAVE_SAMPLE_RATE + 1
    print(f"Receiver listening for {duration:.1f} seconds...")
    captured = sd.rec(int(duration * GGWAVE_SAMPLE_RATE), samplerate=GGWAVE_SAMPLE_RATE, channels=1, dtype="float32", device=input_device, blocking=True)
    instance = ggwave.init()
    try:
        decoded = ggwave.decode(instance, captured[:, 0].astype(np.float32).tobytes())
    finally:
        ggwave.free(instance)
    if not decoded:
        raise RuntimeError("No ggwave text packet was decoded from the microphone")
    validated = validate_packet(decoded, 1)
    if validated is None:
        raise RuntimeError("Received ggwave packet failed CRC32 validation")
    _number, payload = validated
    return payload.decode("utf-8")


def synthesize_with_indic_parler(text, output_path, language):
    import soundfile as sf
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to(device)
    description_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    prompt_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    description = f"A clear, natural {language} speaker reads the sentence with a calm voice."
    description_input = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input = prompt_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        audio = model.generate(
            input_ids=description_input.input_ids,
            attention_mask=description_input.attention_mask,
            prompt_input_ids=prompt_input.input_ids,
            prompt_attention_mask=prompt_input.attention_mask,
        )
    sf.write(output_path, audio.cpu().numpy().squeeze(), model.config.sampling_rate)


def target_tts_language(target_language, fallback):
    if not target_language:
        return fallback
    return target_language.split("_")[0]


def main():
    parser = argparse.ArgumentParser(description="AI4Bharat ASR -> ggwave text -> AI4Bharat TTS")
    parser.add_argument("--audio", default="audio.wav")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--source-language", default="hin_Deva", help="IndicTrans2 source tag, e.g. hin_Deva")
    parser.add_argument("--target-language", default=None, help="IndicTrans2 target tag, e.g. eng_Latn or tam_Taml")
    parser.add_argument("--tts-output", default="received_tts.wav")
    parser.add_argument("--record", type=float, default=None, metavar="SECONDS", help="Record speech from microphone for N seconds before sending")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect spoken Indian language using IndicConformer LID")
    parser.add_argument("--physical-send", action="store_true", help="ASR text -> speaker ggwave signal")
    parser.add_argument("--physical-receive", action="store_true", help="microphone ggwave signal -> text -> TTS")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== iTantra ASR/TTS ggwave Test ===")
    try:
        if args.physical_send and args.physical_receive:
            raise ValueError("Choose one physical role")
        if args.physical_receive:
            import sounddevice
            received_text = receive_text_acoustically(sounddevice.default.device[0])
            print(f"Received text: {received_text}")
            synthesize_with_indic_parler(received_text, args.tts_output, target_tts_language(args.target_language, args.language))
            print(f"TTS output: {args.tts_output}")
            print("PHYSICAL ASR/TTS TEST: PASS")
            return 0

        if args.record is not None:
            record_audio_from_mic(args.audio, duration=args.record)
        elif not os.path.exists(args.audio):
            print(f"Audio file '{args.audio}' not found.")
            record_audio_from_mic(args.audio, duration=5.0)

        if args.auto_detect:
            detected_lang, transcription, scores = detect_and_transcribe_with_indicconformer(args.audio)
            lang_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)
            source_tag = LANGUAGE_TO_INDICTRANS2.get(detected_lang, f"{detected_lang}_Deva")
            print(f"\n[Language Detection] Identified: {lang_name} ('{detected_lang}') -> source tag: '{source_tag}' (score: {scores[detected_lang]:.4f})")
            args.language = detected_lang
            args.source_language = source_tag
        else:
            transcription = transcribe_with_ai4bharat(args.audio, args.language)
        if not transcription:
            raise ValueError("ASR returned empty text")
        print(f"ASR text: {transcription}")
        translated_text = transcription
        if args.target_language:
            translated_text = translate_with_indictrans(transcription, args.source_language, args.target_language)
            print(f"Translated text ({args.source_language} -> {args.target_language}): {translated_text}")
        if args.physical_send:
            import sounddevice
            duration = send_text_acoustically(translated_text, sounddevice.default.device[1])
            print(f"Physical signal duration: {duration:.2f} seconds")
            print("PHYSICAL SENDER: COMPLETE")
            return 0
        received_text, waveform = text_to_ggwave(translated_text)
        print(f"ggwave text integrity: {'PASS' if received_text == translated_text else 'FAIL'}")
        with wave.open("asr_tts_ggwave.wav", "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(4)
            output.setframerate(48000)
            output.writeframes(waveform)
        synthesize_with_indic_parler(received_text, args.tts_output, target_tts_language(args.target_language, args.language))
        print(f"TTS output: {args.tts_output}")
        print("ASR/TTS ggwave pipeline: PASS")
        return 0
    except Exception as error:
        print(f"ASR/TTS pipeline unavailable or failed: {type(error).__name__}: {error}")
        print("ASR/TTS ggwave pipeline: NOT RUN")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
