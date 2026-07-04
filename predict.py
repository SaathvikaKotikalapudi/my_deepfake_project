import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 1. CONFIGURATION ---
MODEL_WEIGHTS = "deepfake_detector.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['FAKE', 'REAL']  # Maps perfectly to your training layout

# --- 2. REBUILD THE NETWORK ARCHITECTURE ---
model = models.mobilenet_v3_small()
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 2)

# Load your trained saved configurations
if os.path.exists(MODEL_WEIGHTS):
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # Set to evaluation mode
    print("✅ Successfully loaded deepfake_detector.pth!")
else:
    print(f"❌ Error: Could not find weights file '{MODEL_WEIGHTS}'")
    exit()

# --- 3. IMAGE PREPROCESSING PIPELINE ---
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Processes an image and outputs the prediction with confidence score."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Open image using PIL
    img = Image.open(image_path).convert('RGB')
    
    # Apply identical training transforms and add a batch dimension
    tensor_img = predict_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_img)
        # Apply Softmax to get percentage probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        prediction = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item() * 100

    print(f"\n🔍 Image: {os.path.basename(image_path)}")
    print(f"🤖 Prediction: {prediction}")
    print(f"📊 Confidence: {score:.2f}%")

if __name__ == "__main__":
    # Test file prompt wrapper
    img_to_test = input("Enter the path/name of an image file to test: ").strip()
    predict_image(img_to_test)