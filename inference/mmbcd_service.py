import os
import sys
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizer
from .config import MMBCD_ROOT, MMBCD_CHECKPOINT


# Path to the MMBCD source code
MMBCD_CODE_PATH = os.path.join(MMBCD_ROOT, "code")


# Add the original MMBCD code to Python's import path
if MMBCD_CODE_PATH not in sys.path:
    sys.path.insert(0, MMBCD_CODE_PATH)


from model import MMBCD


class MMBCDService:

    def __init__(self):
        """
        Initialize the MMBCD service.
        """

        self.model = None

        # Use GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer once
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )

    def load_model(self):
        """
        Load the MMBCD model and trained checkpoint.
        """

        # Don't load the model again if already loaded
        if self.model is not None:
            return

        print(f"Loading MMBCD on {self.device}...")

        # Create MMBCD architecture
        self.model = MMBCD(
            None,   # DINO checkpoint
            0,      # frozen image layers
            224,    # image size
            None,   # RoBERTa checkpoint
            0       # RoBERTa layers to unfreeze
        )

        # Load trained checkpoint
        state_dict = torch.load(
            MMBCD_CHECKPOINT,
            map_location=self.device,
            weights_only=False
        )

        # Checkpoint was trained using DataParallel.
        # Remove "module." prefix from parameter names.
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

        # Load trained weights
        self.model.load_state_dict(state_dict)

        # Move model to CPU/GPU
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("MMBCD loaded successfully.")

    def predict(self, crop_paths, clinical_text):
        """
        Run MMBCD inference using image crops
        and clinical history.
        """

        # Make sure model is loaded
        self.load_model()

        # MMBCD image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Store processed crops
        crop_tensors = []

        # Process every crop
        for crop_path in crop_paths:

            image = Image.open(
                crop_path
            ).convert("RGB")

            tensor = transform(image)

            crop_tensors.append(tensor)

        # Convert list of tensors into one tensor
        # Shape: [number_of_crops, 3, 224, 224]
        crops = torch.stack(crop_tensors)

        # MMBCD expects:
        # [batch_size, number_of_crops, 3, 224, 224]
        #
        # For one mammogram:
        # [1, 5, 3, 224, 224]
        crops = crops.unsqueeze(0)

        # Move image tensor to CPU/GPU
        crops = crops.to(self.device)

        # Tokenize clinical history
        texts = self.tokenizer(
            [clinical_text],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=90
        )

        input_ids = texts["input_ids"].to(
            self.device
        )

        attention_mask = texts["attention_mask"].to(
            self.device
        )

        # Run inference without calculating gradients
        with torch.no_grad():

            logits, _ = self.model(
                crops,
                input_ids,
                attention_mask
            )

            # Convert logits to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Select class with highest probability
            prediction = probabilities.argmax(
                dim=1
            )

        # Return JSON-friendly values
        return {
            "prediction": int(
                prediction.item()
            ),
            "probabilities": [
                float(probabilities[0][0]),
                float(probabilities[0][1])
            ]
        }

    def unload_model(self):
        """
        Release the MMBCD model from memory.
        """

        if self.model is not None:

            del self.model

            self.model = None

            # Clear CUDA memory if GPU is being used
            if self.device == "cuda":
                torch.cuda.empty_cache()

        print("MMBCD unloaded.")