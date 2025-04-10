import os
import torch
import torch.nn as nn
import torch.optim as optim
from classification.data_loader import load_classification_data
from classification.classification_model import get_resnet18_model, train_model
from utils.metrics import evaluate_model


# Configuration

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. Load Data

train_loader, val_loader, test_loader = load_classification_data(
    data_dir='data/classification',
    batch_size=BATCH_SIZE,
    val_split=0.1,
    test_split=0.1
)


# 2. Model 

model = get_resnet18_model(num_classes=2)


# 3. Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 4. Train the Model #

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=DEVICE
)


# 5. Save the Model
# ensure models directory exists
os.makedirs('models', exist_ok=True)

# save the model in the models directory
torch.save(trained_model.state_dict(), 'models/resnet18_brain_tumour.pth')
print("Model saved as resnet18_brain_tumour.pth")


# 6. Evaluate the Model

evaluate_model(model, test_loader, device=DEVICE)

