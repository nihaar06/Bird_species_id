import tensorflow as tf
from pathlib import Path
import numpy as np

class BirdDataset():
    def __init__(self,root_dir):
        self.root_dir=Path(root_dir)
        self.files=[]
        self.label_map={}
        species_folder=sorted(self.root_dir.iterdir())
        for idx,specie in enumerate(species_folder):
            if not specie.is_dir():
                continue
            specie_name=specie.name
            self.label_map[specie_name]=idx
            for file in specie.glob("*.npy"):
                self.files.append(file)

    def load_sample(self,file_path):
        path=Path(file_path)
        specie_name=path.parent.name
        label=self.label_map[specie_name]
        spec=np.load(path)
        # 1. Min-Max Normalization to bring negative dB values to [0, 1]
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 0:
            spec = (spec - min_val) / (max_val - min_val)
        else:
            spec = np.zeros_like(spec) # Handle empty/silent arrays safely
        spec=np.expand_dims(spec,axis=-1)
        return spec.astype(np.float32),label
    
    def get_data(self):
        X=[]
        y=[]
        for file in self.files:
            spec,label=self.load_sample(file)
            X.append(spec)
            y.append(label)
        return np.array(X),np.array(y)


dataset = BirdDataset(
    "dataset/processed/logmel_128"
)
X,y=dataset.get_data()
print(X.shape)
print(y.shape)
from collections import Counter
print(len(y))
print(Counter(y))