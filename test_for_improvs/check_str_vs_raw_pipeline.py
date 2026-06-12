import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.preprocessing.audio_loader import load_audio_pipeline
from src.preprocessing.spectrogram import create_spectrogram
import numpy as np
import librosa


audio, sr = load_audio_pipeline(
    "dataset/raw/train_short_audio/acafly/XC123.ogg"
)

spec = create_spectrogram(audio, sr)

spec = (spec - spec.min()) / (spec.max() - spec.min())

print("RAW:")
print(spec.shape)
print(spec.min())
print(spec.max())
print(spec.mean())

def process_field_audio(file_path, sr=32000, duration=5, top_db=45):
    # Load raw sound clip signals natively
    audio, _ = librosa.load(file_path, sr=sr)
    
    # Rescale peak waveform variance amplitudes
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
        
    # Apply wilderness field recording silence threshold (top_db=45 for faint calls)
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    
    # Apply your lightning-fast Vectorized Local Energy focus window locator
    target_length = sr * duration
    if len(audio) > target_length:
        frame_length, hop_length = 2048, 512
        num_frames = (len(audio) - frame_length) // hop_length + 1
        
        # Stride matrix view mappings to replace slow python processing loops
        shape = (num_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        audio_windows = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        energy = np.sum(audio_windows**2, axis=1)
        
        if len(energy) > 0:
            max_energy_idx = np.argmax(energy)
            center_sample = max_energy_idx * hop_length
            start = max(0, center_sample - (target_length // 2))
            end = start + target_length
            if end > len(audio):
                end = len(audio)
                start = max(0, end - target_length)
            audio = audio[start:end]
        else:
            audio = audio[:target_length]
    else:
        # Pad shorter sound cues safely with silent trailing zeroes
        audio = np.pad(audio, (0, target_length - len(audio)))
        
    # Generate the identical 128-bin Mel Spectrogram representation matrix
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Min-Max scaling mapped to [0, 255] for EfficientNetB0 expectations (CRITICAL)
    min_val, max_val = log_mel.min(), log_mel.max()
    if max_val - min_val > 0:
        log_mel = ((log_mel - min_val) / (max_val - min_val)) * 255.0
    else:
        log_mel = np.zeros_like(log_mel)
        
    # Triplicate matrix planes into 3 matching data channels (RGB format for EfficientNet)
    log_mel = np.expand_dims(log_mel, axis=-1)
    log_mel = np.repeat(log_mel, 3, axis=-1)
    
    # Return batch array wrapper shape (1, 128, 313, 3)
    return np.expand_dims(log_mel, axis=0)


print("STREAMLIT")
tensor = process_field_audio(
    "dataset/raw/train_short_audio/acafly/XC123.ogg"
)

print(tensor.shape)
print(tensor.min())
print(tensor.max())
print(tensor.mean())