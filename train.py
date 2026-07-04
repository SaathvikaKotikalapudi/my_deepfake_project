import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset"

print(f"Using device: {DEVICE}")

# --- 2. DATA AUGMENTATION & PIPELINES ---
# Normalizing values matched to standard ImageNet training baselines
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Automatically loading images using the folder structure we just generated
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}

# Change this line in train.py:
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
               for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(f"Detected Classes: {class_names} -> Mapped as {image_datasets['train'].class_to_idx}")

# --- 3. THE MODEL ARCHITECTURE (Transfer Learning) ---
# Leveraging MobileNetV3 for a fast, resource-friendly balance of high performance
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Adjust classification head for binary prediction (REAL vs FAKE)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 2) 
model = model.to(DEVICE)

# --- 4. LOSS & OPTIMIZATION SETTINGS ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. TRAINING LOOP ---
print("\n🚀 Starting Training Loop...")
for epoch in range(EPOCHS):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"Epoch {epoch+1}/{EPOCHS} | {phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# --- 6. SAVE MODEL WEIGHTS ---
torch.save(model.state_dict(), "deepfake_detector.pth")
print("\n✅ Training complete! Weights saved as 'deepfake_detector.pth'")