import os
import sys
import torch
import soundfile as sf
from encodec import EncodecModel
from encodec.utils import convert_audio

# Force UTF-8 encoding for stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("STEP 2: Meta EnCodec Neural Audio Compression Test")
    print("=" * 60)

    # 1. Load original audio file
    audio_path = "audio.wav"
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found!")
        return

    orig_file_size = os.path.getsize(audio_path)
    data, orig_sr = sf.read(audio_path, dtype="float32")
    
    # Calculate audio duration and channel stats
    if data.ndim == 1:
        num_samples = len(data)
        num_channels = 1
    else:
        num_samples = data.shape[0]
        num_channels = data.shape[1]
    
    duration = num_samples / orig_sr
    raw_pcm_bytes = num_samples * num_channels * 2  # 16-bit PCM equivalent size

    print(f"\n[1] Original Audio Properties:")
    print(f"    - File Path: {audio_path}")
    print(f"    - File Size: {orig_file_size / 1024:.2f} KB ({orig_file_size} bytes)")
    print(f"    - Sample Rate: {orig_sr} Hz")
    print(f"    - Channels: {num_channels}")
    print(f"    - Duration: {duration:.2f} seconds")
    print(f"    - Raw PCM Size (16-bit): {raw_pcm_bytes / 1024:.2f} KB")

    # 2. Load Pretrained EnCodec 24kHz Model
    print(f"\n[2] Loading Pretrained Meta EnCodec 24kHz Model...")
    model = EncodecModel.encodec_model_24khz()
    model.eval()
    
    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert numpy audio to torch tensor (shape: batch_size=1, channels=1, samples)
    wav = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

    # Convert audio to model target sample rate (24kHz) and mono channel (1)
    wav_24k = convert_audio(wav, orig_sr, model.sample_rate, model.channels)

    supported_bandwidths = model.target_bandwidths
    print(f"    - Model Sample Rate: {model.sample_rate} Hz")
    print(f"    - Supported Bitrates: {supported_bandwidths} kbps")

    # 3. Test Bitrates: 24.0, 12.0, 6.0, 3.0, 1.5 kbps
    test_bitrates = [24.0, 12.0, 6.0, 3.0, 1.5]
    print(f"\n[3] Testing Compression & Reconstruction across Bitrates:")
    print("-" * 60)

    for bw in test_bitrates:
        if bw not in supported_bandwidths:
            print(f"Skipping unsupported bitrate: {bw} kbps")
            continue

        model.set_target_bandwidth(bw)
        
        with torch.no_grad():
            # Encode audio into discrete VQ codes
            encoded_frames = model.encode(wav_24k)
            
            # Decode VQ codes back into audio waveform
            decoded_wav = model.decode(encoded_frames)

        # Calculate compressed code size
        total_code_elements = 0
        code_shapes = []
        for codes, scale in encoded_frames:
            total_code_elements += codes.numel()
            code_shapes.append(tuple(codes.shape))

        # Each code is an index up to 1024 (10 bits per code)
        compressed_bits = total_code_elements * 10
        compressed_bytes = compressed_bits // 8
        compression_ratio = raw_pcm_bytes / compressed_bytes if compressed_bytes > 0 else 0

        bw_str = f"{int(bw)}" if bw.is_integer() else f"{bw}"
        out_filename = f"reconstructed_{bw_str}kbps.wav"
        
        # Prepare numpy array for saving with soundfile
        decoded_audio_np = decoded_wav.squeeze(0).squeeze(0).cpu().numpy()
        sf.write(out_filename, decoded_audio_np, model.sample_rate)
        
        reconstructed_file_size = os.path.getsize(out_filename)

        print(f"Bitrate: {bw:>4.1f} kbps | Code Shape: {code_shapes[0]} | Codebooks: {code_shapes[0][1]}")
        print(f"  -> Compressed Code Size: {compressed_bytes} bytes ({compressed_bytes/1024:.2f} KB)")
        print(f"  -> Compression Ratio: {compression_ratio:.1f}x reduction vs raw PCM")
        print(f"  -> Saved Output: {out_filename} ({reconstructed_file_size / 1024:.2f} KB)")
        print(f"  -> Status: SUCCESS - Reconstructed audio generated and playable\n")

    print("=" * 60)
    print("EnCodec Step 2 Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
