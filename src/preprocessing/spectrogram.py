import librosa
import numpy as np
from pathlib import Path

#CREATE MEL_SPEC
def create_spectrogram(audio,sr,n_mels=128):
    mel_spec=librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    log_mel_spec=librosa.power_to_db(
        mel_spec,
        ref=np.max
    )
    return log_mel_spec

#SAVE SPECTOGRAM
def save_spectrogram(spec,save_path):
    save_path=Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    save_path=save_path.with_suffix(".npy")
    np.save(save_path,spec.astype(np.float32))

#PIPELINE
def spectogram_pipeline(audio,save_path,sr=32000):
    log_mel_spec=create_spectrogram(audio,sr)
    save_spectrogram(log_mel_spec,save_path)
    return log_mel_spec