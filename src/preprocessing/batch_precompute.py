
from pathlib import Path
print(Path.cwd())
from audio_loader import load_audio_pipeline
from spectrogram import spectogram_pipeline
import numpy as np

def batch_preprocess():
    RAW_DATASET=Path("dataset/raw/train_short_audio")
    PROCESSED=Path("dataset/processed/logmel_128")
    for species_folder in list(RAW_DATASET.iterdir())[:1]:
        if not species_folder.is_dir():
            continue
        species_name=species_folder.name
        print(f"\nProcessing {species_name}...")
        for specie_audio in list(species_folder.glob("*.ogg"))[:2]:
            output_folder=PROCESSED/species_name
            output_file=output_folder/f"{specie_audio.stem}.npy"
            if output_file.exists():
                continue
            audio,sr=load_audio_pipeline(specie_audio)
            spectogram_pipeline(audio,output_file,sr)
            print(f"Saved:{output_file.name}")
    print("Done!")

batch_preprocess()