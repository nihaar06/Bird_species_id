import torch
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir=Path(root_dir)
        self.label_map={}
        self.files=[]
        species_folder=sorted(self.root_dir.iterdir())
        for i,specie_folder in enumerate(species_folder):
            if not specie_folder.is_dir():
                continue
            self.label_map[specie_folder.name]=i
            for file_path in specie_folder.glob("*.npy"):
                self.files.append(file_path)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path=self.files[idx]
        spec=np.load(file_path)
        tensor=torch.tensor(
            spec,
            dtype=torch.float32
        )
        tensor=tensor.unsqueeze(0)
        species_name=file_path.parent.name
        label=self.label_map[species_name]
        return tensor,label

dataset = BirdDataset(
    "dataset/processed/logmel_128"
)

print(len(dataset))

spec, label = dataset[0]

print(spec.shape)
print(label)