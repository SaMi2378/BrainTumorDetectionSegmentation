import matplotlib.pyplot as plt
import os

# --- CLASSIFICATION VISUALISATION ---
def save_training_plots(train_losses, val_losses, train_accs, val_accs, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

# --- SEGMENTATION VISUALISATION ---

import torch
import matplotlib.pyplot as plt
import os

def visualise_prediction(model, dataset, device='cuda', sample_idx=0, save_path="models/sample_segmentation.png"):
    model.eval()
    sample = dataset[sample_idx]

    # Handle dataset with or without ground truth mask
    if isinstance(sample, tuple):
        image, mask = sample
        has_mask = True
        mask = mask.cpu().numpy()
    else:
        image = sample
        has_mask = False

    image = image.unsqueeze(0).to(device)  # Shape: [1, C, D, H, W]

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Shape: [D, H, W]

    # Select central slice
    mid_slice = pred.shape[0] // 2
    modality = image.squeeze()[0].cpu().numpy()  # Use first modality (T1c)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3 if has_mask else 2, 1)
    plt.imshow(modality[mid_slice], cmap='gray')
    plt.title("Input (Mid Slice)")
    plt.axis('off')

    if has_mask:
        plt.subplot(1, 3, 2)
        plt.imshow(mask[mid_slice], cmap='viridis')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred[mid_slice], cmap='viridis')
        plt.title("Predicted Mask")
        plt.axis('off')
    else:
        plt.subplot(1, 2, 2)
        plt.imshow(pred[mid_slice], cmap='viridis')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

    print(f"✅ Saved visualisation at: {save_path}")
