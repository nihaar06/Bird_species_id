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
def trim_pad(audio,sr,duration=5):
    target_length=sr*duration
    #TRIM if original audio longer
    if len(audio)>target_length:
        start=(len(audio)-target_length)//2
        audio=audio[start:start+target_length]
    #PAD if original audio smaller
    else:
        pad_length=target_length-len(audio)
        audio=np.pad(audio,(0,pad_length))
    return audio

def load_audio_pipeline(file_path,sr=32000,top_db=20,duration=5):
    audio,sr=load_audio(file_path,sr)
    audio=normalize_audio(audio)
    audio=remove_silence(audio,top_db)
    audio=trim_pad(audio,sr,duration)
    return audio,sr
