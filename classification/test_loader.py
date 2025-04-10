from data_loader import load_classification_data

train_loader, val_loader, test_loader = load_classification_data(
    data_dir="data/classification",
    image_size=224,
    batch_size=16
)

# Check a batch
for images, labels in train_loader:
    print("Batch of images shape:", images.shape)
    print("Batch of labels:", labels)
    break
