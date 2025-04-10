import torch
from torchvision import transforms
from PIL import Image
import random
import os
import sys

# Ensure root directory (BrainTumorDetectionSegmentation/) is in the path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from classification.classification_model import get_resnet18_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = get_resnet18_model(num_classes=2)
model.load_state_dict(torch.load('models/resnet18_brain_tumour.pth', map_location=device))
model.eval()
model.to(device)

# Choose random image from yes/no folders
base_dir = 'data/classification'
class_dirs = ['yes', 'no'] # yes = tumour, no = no tumour
chosen_class = random.choice(class_dirs) # randomly choose a class
image_file = random.choice(os.listdir(os.path.join(base_dir, chosen_class))) # randomly choose an image from the chosen class
image_path = os.path.join(base_dir, chosen_class, image_file) # create the path to the image

# Preprocess with RGB conversion
image = Image.open(image_path).convert("RGB")  # Convert to 3 channels

transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224
    transforms.ToTensor(), # convert the image to a tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # For 3 channels
])
# Add batch dimension
image = transform(image).unsqueeze(0).to(device)

# Predict
output = model(image) # pass the image through the model
pred = torch.argmax(output, dim=1).item() # get the predicted class
# Print the prediction
print(f"✅ Image: {image_file} | Ground Truth: {chosen_class.upper()} | Predicted: {'TUMOUR' if pred == 1 else 'NO TUMOUR'}")
