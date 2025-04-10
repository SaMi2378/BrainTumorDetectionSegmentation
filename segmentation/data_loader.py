import torch
from torch.utils.data import DataLoader, random_split
from segmentation.segmentation_dataset import BraTSSegmentationDataset

def get_segmentation_loaders(
    data_dir, 
    batch_size=2, 
    val_split=0.1, 
    test_split=0.1, 
    target_shape=(128, 128, 128)
):
    full_dataset = BraTSSegmentationDataset(data_dir, target_shape=target_shape)
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
