
import os
import torch
import torch.nn as nn
import torch.optim as optim
from segmentation.data_loader import get_segmentation_loaders
from segmentation.segmentation_model import get_segmentation_model
from utils.metrics import dice_score, evaluate_segmentation_model
from utils.visualisations import visualise_prediction
from segmentation.segmentation_dataset import BraTSSegmentationDataset
from segmentation.Seg_Val_dataset import BraTSSegmentationDataset2
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = 'data/segmentation/train'
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Data ---
train_loader, val_loader, test_loader = get_segmentation_loaders(DATA_DIR, batch_size=BATCH_SIZE)

# --- Model, Loss, Optimizer ---
model = get_segmentation_model(in_channels=3, num_classes=4).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Train Loss: {running_loss/len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
dice_total = 0.0
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Validation"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        dice_total += dice_score(preds, masks)

print(f"Validation Dice Score: {dice_total/len(val_loader):.4f}")

# --- Save Model ---
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/unet3d_brats2025.pth')
print("Model saved to models/unet3d_brats2025.pth")

# --- Evaluate Model ---    
evaluate_segmentation_model(model, val_loader, device=DEVICE, num_classes=4)

# --- Visualise Prediction ---
# Load single sample from train or val folder (not DataLoader)
val_dataset = BraTSSegmentationDataset2("data/segmentation/val",)
visualise_prediction(model, val_dataset, device=DEVICE)