import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

class BraTSSegmentationDataset2(Dataset):
    def __init__(self, data_dir, transform=None, target_shape=(128, 128, 128)):
        self.data_dir = data_dir
        self.subjects = sorted(os.listdir(data_dir))
        self.transform = transform
        self.target_shape = target_shape
        

    def __len__(self):
        return len(self.subjects)

    def load_nifti(self, filepath):
        img = nib.load(filepath)
        img_data = img.get_fdata()
        return img_data

    def preprocess(self, image):
        # Normalize intensity
        image = (image - np.mean(image)) / np.std(image)
        
        # Resize to uniform shape
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        image = F.interpolate(image, size=self.target_shape, mode='trilinear', align_corners=False)
        return image.squeeze(0)  # [1, D, H, W]



    def __getitem__(self, idx):
        subject_dir = os.path.join(self.data_dir, self.subjects[idx])
        subject_id = self.subjects[idx]

        # Load modalities
        t1c = self.load_nifti(os.path.join(subject_dir, f"{self.subjects[idx]}-t1c.nii.gz"))
        t2 = self.load_nifti(os.path.join(subject_dir, f"{self.subjects[idx]}-t2w.nii.gz"))
        flair = self.load_nifti(os.path.join(subject_dir, f"{self.subjects[idx]}-t2f.nii.gz"))

     
        # Stack modalities (T1c, T2, FLAIR)
        image = np.stack([t1c, t2, flair], axis=0)
        image = (image - np.mean(image)) / np.std(image)  # Normalize intensity
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension: [1, 3, D, H, W]
        image = F.interpolate(image, size=self.target_shape, mode='trilinear', align_corners=False)
        image = image.squeeze(0)  # Remove batch dimension: [3, D, H, W]

       
        

        if self.transform:
            image = self.transform(image)

        return image.float()

  
