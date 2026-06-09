import librosa
import numpy as np
from pathlib import Path

#LOAD AUDIO
def load_audio(file_path,sr=32000):
    audio,sr=librosa.load(file_path,sr=sr)
    return audio,sr

#NORMALIZE
def normalize_audio(audio):
    max_val=np.max(np.abs(audio))
    if max_val>0:
        audio=audio/max_val
    return audio

#REMOVE SILENCE 
# -since we are trimming the audio to 5 seconds, removing silence can help to ensure that we are capturing the relevant audio content
#  and not just silence, which can improve the quality of the spectrograms and the performance of the model.
def remove_silence(audio,top_db=30):
    audio,_=librosa.effects.trim(audio,top_db=top_db)
    return audio

#TRIM_PAD
def trim_pad(audio, sr, duration=5):
    target_length = sr * duration
    
    if len(audio) > target_length:
        frame_length = 2048
        hop_length = 512
        
        # 1. Fast Vectorized Energy Calculation via NumPy strides
        # This acts exactly like a sliding window but processes instantly in memory
        num_frames = (len(audio) - frame_length) // hop_length + 1
        shape = (num_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        
        # Create the sliding views without duplication
        audio_windows = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        
        # Calculate energy across all windows instantly
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
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length))
        
    return audio

def load_audio_pipeline(file_path,sr=32000,top_db=45,duration=5):
    audio,sr=load_audio(file_path,sr)
    audio=normalize_audio(audio)
    audio=remove_silence(audio,top_db)
    audio=trim_pad(audio,sr,duration)
    return audio,sr
