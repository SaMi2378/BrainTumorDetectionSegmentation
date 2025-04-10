import os
import glob
from sklearn.model_selection import train_test_split
from classification.custom_dataset import BrainTumourDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_classification_data(data_dir, image_size=224, batch_size=16, val_split=0.1, test_split=0.1):
    yes_dir = os.path.join(data_dir, "yes")
    no_dir = os.path.join(data_dir, "no")

    # Extensions to support
    valid_exts = ["*.jpg", "*.jpeg", "*.JPG", "*.png"]

    # Collect file paths
    yes_files = []
    for ext in valid_exts:
        yes_files.extend(glob.glob(os.path.join(yes_dir, ext)))

    no_files = []
    for ext in valid_exts:
        no_files.extend(glob.glob(os.path.join(no_dir, ext)))

    all_files = yes_files + no_files
    all_labels = [1] * len(yes_files) + [0] * len(no_files)

    # Stratified Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_files, all_labels, test_size=test_split, stratify=all_labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split, stratify=y_trainval, random_state=42
    )  # ~10% val   

    # Define Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Grayscale: 3 channel mean/std for model compatibility
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Datasets
    train_ds = BrainTumourDataset(X_train, y_train, transform=train_transform)
    val_ds = BrainTumourDataset(X_val, y_val, transform=val_test_transform)
    test_ds = BrainTumourDataset(X_test, y_test, transform=val_test_transform)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
