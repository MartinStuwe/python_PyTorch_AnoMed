import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import pandas as pd

from datasets import load_dataset
from PIL import Image

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_dir = "./data"
celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)

df = celeba.to_pandas()

attr_names = df.columns[1:-1]
image_col_name = df.columns[0]

attributes = df[attr_names].values.astype(np.float32)

inputs = df[image_col_name].values

targets = (attributes + 1) / 2

train_percentage = 0.8
val_percentage = 0.1
test_percentage = 1 - train_percentage - val_percentage

training_amount = int(train_percentage * len(inputs))
validation_amount = int(val_percentage * len(inputs))
testing_amount = int(train_percentage * len(inputs))

train_inputs = inputs[0:training_amount]
val_inputs = inputs[training_amount:training_amount+validation_amount]
test_inputs = inputs[training_amount+validation_amount:]


train_targets = targets[0:training_amount]
val_targets = targets[training_amount:training_amount+validation_amount]
test_targets = targets[training_amount+validation_amount:]

train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
val_targets = torch.tensor(val_targets, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)


from PIL import Image
from io import BytesIO

# Define custom transform
custom_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

from torch.utils.data import DataLoader, Dataset

class ViTDataset(Dataset):
    def __init__(self, images, targets, feature_extractor):
        self.images = images  # List of image paths or byte data dictionaries
        self.targets = targets  # Corresponding targets
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_data = self.images[idx]

        if isinstance(image_data, dict):  # If image data is stored as a dict (e.g., {'bytes': ...})
            img_bytes = image_data['bytes']
            image = Image.open(BytesIO(img_bytes))
        else:  # Otherwise, assume it's a file path
            image = Image.open(image_data)

        # Apply feature extractor
        encoded_input = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_input['pixel_values'].squeeze(0)  # Remove the batch dimension

        target = self.targets[idx]
        return pixel_values, target
    

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=1):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits  # Access logits directly
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        eval_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images).logits  # Access logits directly
                loss = criterion(outputs, targets)
                eval_loss += loss.item()

                predictions = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                predicted_labels = (predictions > 0.5).float()  # Convert to binary labels

                correct_predictions += (predicted_labels == targets).sum().item()
                total_predictions += targets.numel()

        avg_eval_loss = eval_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")

        scheduler.step()

    return model

from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load the pre-trained ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=40)
vit_model.to(device)

# Create datasets and dataloaders
vit_train_dataset = ViTDataset(train_inputs, train_targets, feature_extractor)
vit_val_dataset = ViTDataset(val_inputs, val_targets, feature_extractor)

vit_train_loader = DataLoader(vit_train_dataset, batch_size=16, shuffle=True)
vit_val_loader = DataLoader(vit_val_dataset, batch_size=16, shuffle=False)

# Optimizer and scheduler
optimizer = optim.AdamW(vit_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = ExponentialLR(optimizer, gamma=0.99)

# Train the ViT model
trained_vit_model = train_model(vit_model, vit_train_loader, vit_val_loader, nn.BCEWithLogitsLoss(), optimizer, scheduler)

# Save the trained ViT model
torch.save(trained_vit_model.state_dict(), "./trained/AnoMed/vit_model.pth")
