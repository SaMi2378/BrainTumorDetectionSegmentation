# 🧠 Brain Tumour Detection & Segmentation with Deep Learning

This project aims to implement and evaluate deep learning models for **brain tumour classification** and **tumour segmentation** using MRI scans. The pipeline includes:

- Binary classification using **ResNet18**
- Multi-class segmentation using **3D U-Net**
- Evaluation metrics (accuracy, precision, recall, F1-score, AUC, Dice)
- Visualisation (confusion matrix, loss/accuracy curves, segmentation overlays)

---

## 📂 Project Structure

BrainTumorDetectionSegmentation/
│
├── classification/                     # Classification-related code
│   ├── __init__.py
│   ├── classification_model.py         # ResNet18 model + training logic
│   ├── data_loader.py                  # Data loading for classification
│   └── test_loader.py
│
├── segmentation/                       # Segmentation-related code
│   ├── __init__.py
│   ├── segmentation_model.py           # 3D U-Net model
│   ├── segmentation_dataset.py         # Dataset for training & validation
│   ├── seg_val_dataset.py              # Dataset for val-only without GT
│   └── data_loader.py                  # Data loading for segmentation
│
├── scripts/                            # Sample prediction scripts
│   ├── Classification_Script.py        # Test classification on random image
│   └── Segmentation_Script.py          # Predict & visualise segmentation
│
├── utils/                              # Helper and visualisation utilities
│   ├── __init__.py
│   ├── metrics.py                      # Metrics for classification & dice
│   ├── visualisations.py               # Segmentation visualisation function
│   └── existing.py                     # Older or alternative utility scripts
│
├── notebooks/                          # Jupyter demo notebooks
│   ├── classification_demo.ipynb       # Inference demo for classification
│   └── segmentation_demo.ipynb         # Inference demo for segmentation
│
├── models/                             # Saved model weights and figures
│   ├── resnet18_brain_tumour.pth
│   ├── unet3d_brain_tumour.pth
│   ├── confusion_matrix.png
│   └── sample_segmentation.png
│
├── data/                               # Dataset folders (not included)
│   ├── classification/                 # yes/ and no/ folders of MRIs
│   └── segmentation/                   # BraTS-like structured folders
│       ├── train/                      # Full data with GT
│       └── val/                        # Data without seg.nii.gz
│
├── main.py                             # Train classification model
├── segmentation_main.py                # Train segmentation model
├── README.md                           # Project description
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Excludes models/, __pycache__/, etc.



---

## 💻 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt

Train ResNet18 on MRI scans to detect tumour presence:
python main.py

Evaluate a random sample:
python scripts/Classification_Script.py


Segmentation
Train a 3D U-Net on BraTS dataset for tumour region segmentation:
python segmentation_main.py


Visualise prediction from validation set:
python scripts/Segmentation_Script.py



Developed by Sami Ullah, 2025
For final year Software Engineering project


---

Let me know if you'd like it tailored for GitHub Pages, a portfolio, or with badges.