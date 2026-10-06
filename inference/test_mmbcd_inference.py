import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizer

# Add MMBCD code folder to Python path
sys.path.insert(0, r"C:\Users\ravis\Desktop\MMBCD-main\code")

from model import MMBCD


# -----------------------------
# Paths
# -----------------------------
CHECKPOINT = r"C:\Users\ravis\Desktop\MMBCD-main\models\mmbcd\mmbcd_best.pt"

CROPS_DIR = r"C:\Users\ravis\Desktop\mammogram_data\mmbcd_crops"


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -----------------------------
# Load MMBCD model
# -----------------------------
print("Loading MMBCD model...")

model = MMBCD(
    None,   # DINO checkpoint
    0,      # frozen layers
    224,    # image size
    None,   # RoBERTa checkpoint
    0       # RoBERTa layers to unfreeze
)

# Load trained MMBCD checkpoint
state = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

# Remove "module." prefix from DataParallel checkpoint
state = {
    key.replace("module.", "", 1): value
    for key, value in state.items()
}

model.load_state_dict(state)

model.to(device)
model.eval()

print("MMBCD model loaded successfully.")


# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# -----------------------------
# Load the 5 crops
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

crop_tensors = []

for i in range(1, 6):

    path = os.path.join(
        CROPS_DIR,
        f"crop_{i}.png"
    )

    image = Image.open(path).convert("RGB")

    tensor = transform(image)

    crop_tensors.append(tensor)

# Shape: [5, 3, 224, 224]
crops = torch.stack(crop_tensors)

# MMBCD expects:
# [batch_size, number_of_crops, channels, height, width]
crops = crops.unsqueeze(0)

print("Crops tensor shape:", crops.shape)


# -----------------------------
# Clinical text
# -----------------------------
clinical_text = "bl mastalgia 2 months"

texts = tokenizer(
    [clinical_text],
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=90
)

input_ids = texts["input_ids"].to(device)
attention_mask = texts["attention_mask"].to(device)


# -----------------------------
# MMBCD inference
# -----------------------------
crops = crops.to(device)

print("Running MMBCD inference...")

with torch.no_grad():

    logits, _ = model(
        crops,
        input_ids,
        attention_mask
    )

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    prediction = probabilities.argmax(
        dim=1
    )

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Prediction:", prediction)


print()
print("MMBCD inference completed.")