import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.visualisations import save_training_plots


# --- Model Definition ---
def get_resnet18_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # Replaces the final layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=15, device='cuda'):
    model.to(device)

    # Initialize lists to store training/validation loss and accuracy
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Accuracy: {train_acc:.2f}%")

        # Validation
        val_acc = validate_model(model, val_loader, device)
        if scheduler:
            scheduler.step()

        # Store metrics
        train_losses.append(running_loss / len(train_loader))
        train_accs.append(train_acc)

        # Validate and get loss
        val_loss, val_acc = validate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    # Save training plots
    save_training_plots(train_losses, val_losses, train_accs, val_accs)

    return model

# --- Validation Function ---
def validate_model(model, val_loader, device='cuda', criterion=None):
    model.eval() # Set model to evaluation mode
    correct = 0 # Initialize correct predictions
    total = 0 # Initialize total number of samples
    val_loss = 0.0 # Initialize validation loss

    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"): # Iterate over validation loader
            images, labels = images.to(device), labels.to(device) # Move images and labels to device
            outputs = model(images) # Get model predictions

            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}% | Val Loss: {avg_loss:.4f}")
    return avg_loss, accuracy

