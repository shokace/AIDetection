import torch
from torchvision import models, transforms
from PIL import Image

from config import MODEL_PATH, INFER_IMAGE_PATH, CLASS_NAMES
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(INFER_IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

print(f"Prediction: {CLASS_NAMES[predicted.item()].upper()}")
print(f"Confidence: {confidence:.4f}")