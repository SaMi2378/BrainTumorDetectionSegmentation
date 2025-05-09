import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BrainTumourDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("L")  # ensuring images are grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
