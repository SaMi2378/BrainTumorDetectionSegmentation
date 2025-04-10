import os
import sys
import torch
import matplotlib.pyplot as plt
# --- Ensure Root Directory is in Path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from segmentation.segmentation_model import get_segmentation_model
from segmentation.Seg_Val_dataset import BraTSSegmentationDataset2  # Uses only modalities, not seg.nii.gz


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/unet3d_brats2025.pth"
DATA_DIR = "data/segmentation/val"
SAVE_PATH = "models/sample_segmentation_val_only.png"

# --- Load Dataset ---
val_dataset = BraTSSegmentationDataset2(data_dir=DATA_DIR, target_shape=(128, 128, 128))
image = val_dataset[0].unsqueeze(0).to(DEVICE)  # Shape: [1, C, D, H, W]

# --- Load Trained Model ---
model = get_segmentation_model(in_channels=3, num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Generate Prediction ---
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# --- Prepare Visualisation ---
mid_slice = image.shape[2] // 2  # Depth-wise middle slice
input_modality = image.squeeze()[0].cpu().numpy()  # First modality (T1c)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(input_modality[mid_slice], cmap="gray")
plt.title("Input Image (Mid Slice)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(prediction[mid_slice], cmap="viridis")
plt.title("Predicted Mask (Mid Slice)")
plt.axis("off")

plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH)
plt.show()
print(f"✅ Prediction visualisation saved to: {SAVE_PATH}")
