import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# Paths
image_path = "./../test/00000001_000.png"
gt_csv = "./../data/raw/Data_Entry_2017.csv"

# Classes
classes = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Model paths
models_info = {
    "ResNet50_Frozen": "./../models/resnet50_frozen_6_best.pt",
    "ResNet50_Finetuned": "./../models/resnet50_finetune_best.pt",
    "DenseNet121_Frozen": "./../models/densenet_frozen.pt",
    "DenseNet121_Finetuned": "./../models/densenet_finetuned.pt"
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Load models
model_objects = {}
for name, path in models_info.items():
    if "resnet50" in name.lower():
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 14)
    else:
        model = models.densenet121(pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 14)
    
    checkpoint = torch.load(path, map_location=device)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    model_objects[name] = model

# Get ground truth labels
gt_df = pd.read_csv(gt_csv)
gt_row = gt_df[gt_df['Image Index'] == os.path.basename(image_path)]

if gt_row.empty:
    true_labels = [0] * 14
else:
    labels_str = gt_row['Finding Labels'].values[0]
    true_labels = [1 if c in labels_str.split('|') else 0 for c in classes]

# Make predictions
data = {'Class': classes, 'True': true_labels}

for name, model in model_objects.items():
    with torch.no_grad():
        outputs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]
    data[name] = outputs

# Create and display DataFrame
df = pd.DataFrame(data)
print(df)