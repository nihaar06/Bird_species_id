from torch.utils.data import DataLoader
from dataset.bird_dataset import BirdDataset
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential

dataset=BirdDataset("dataset/processed/logmel_128")
dataloader=DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)