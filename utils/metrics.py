import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Tumour", "Tumour"], yticklabels=["No Tumour", "Tumour"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()

    return acc, prec, rec, f1, auc


# Dice Score
# def dice_score(preds, targets, smooth=1e-5):
#     """
#     Computes Dice Score for multi-class segmentation with one-hot or class labels.
#     Both preds and targets should be [B, C, D, H, W] or [B, D, H, W] with class labels.

#     Args:
#         preds: predicted logits or class labels
#         targets: ground truth labels
#         smooth: smoothing factor to avoid division by zero

#     Returns:
#         Dice score (float)
#     """
#     # Convert to one-hot encoding if not already
#     if preds.shape != targets.shape:
#         preds = torch.argmax(preds, dim=1)

#     preds = preds.float()
#     targets = targets.float()

#     # Compute intersection and union
#     intersection = (preds * targets).sum()
#     union = preds.sum() + targets.sum()
#     dice = (2. * intersection + smooth) / (union + smooth)
#     # Compute Dice score for each class
#     return dice.item()

# --- SEGMENTATION METRICS ---
def dice_score(preds, targets, num_classes=4, epsilon=1e-6):
    """
    Computes multi-class Dice score averaged over all classes.
    preds: logits or class indices, shape [B, C, D, H, W] or [B, D, H, W]
    targets: ground truth, shape [B, D, H, W]
    """
    if preds.shape != targets.shape:
        preds = torch.argmax(preds, dim=1)

    dice_total = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_total += dice

    return (dice_total / num_classes).item()


def evaluate_segmentation_model(model, dataloader, device='cuda', num_classes=4):
    model.eval()
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_dice = dice_score(outputs, masks, num_classes=num_classes)
            total_dice += batch_dice
            num_batches += 1

    avg_dice = total_dice / num_batches
    print(f"\nAverage Dice Score: {avg_dice:.4f}")
    return avg_dice
