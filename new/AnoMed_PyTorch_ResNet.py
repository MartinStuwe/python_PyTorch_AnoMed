import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageFile
import os
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup_ddp():
    dist.destroy_process_group()

# Define custom dataset and DataLoader
class CelebADataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images  # List of image paths or byte data dictionaries
        self.targets = targets  # Corresponding targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_data = self.images[idx]

        # If the image data is a dict, extract the 'bytes' field and load it
        if isinstance(image_data, dict) and 'bytes' in image_data:
            img_bytes = image_data['bytes']
            image = Image.open(BytesIO(img_bytes))
        else:
            # Otherwise, assume it's a file path and open it directly
            image = Image.open(image_data)

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]
        return image, target


# Function to train the model using DDP
def train_ddp(rank, world_size, train_inputs, train_targets, val_inputs, val_targets, num_epochs=100, batch_size=16):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Define transformations
    custom_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Prepare datasets and distributed samplers
    train_dataset = CelebADataset(train_inputs, train_targets, transform=custom_transform)
    val_dataset = CelebADataset(val_inputs, val_targets, transform=custom_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)  # No sampler for validation


    # Load a pre-trained ResNet model
    resnet_model = models.resnet18(weights='DEFAULT')
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, 40)  # Adjust for 40 attributes

    # Move the model to the appropriate device
    resnet_model = resnet_model.to(device)
    ddp_model = DDP(resnet_model, device_ids=[rank])

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=3e-5, weight_decay=1e-1)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # Only rank 0 initializes the SummaryWriter
    if rank == 0:
        writer = SummaryWriter(log_dir="./tensorboard_logs/resnet/")

    # Training loop
    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch

        # Reset training accumulators
        total_train_loss = 0.0
        correct_predictions_train = 0
        total_predictions_train = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            total_train_loss += loss.item() * images.size(0)  # Multiply by batch size
            predictions = torch.sigmoid(outputs)
            predicted_labels = (predictions > 0.5).float()
            correct_predictions_train += (predicted_labels == targets).sum().item()
            total_predictions_train += targets.numel()

        # Calculate average training loss and accuracy
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_accuracy = correct_predictions_train / total_predictions_train

        dist.barrier()
        #if rank == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")


        # Validation after each epoch
        ddp_model.eval()

        # Reset validation accumulators
        total_eval_loss = 0.0
        correct_predictions_val = 0
        total_predictions_val = 0
# TODO: Debug step
# 
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                val_outputs = ddp_model(val_images)
                val_loss = criterion(val_outputs, val_targets)
                total_eval_loss += val_loss.item() * val_images.size(0)  # Multiply by batch size

                val_predictions = torch.sigmoid(val_outputs)
                val_predicted_labels = (val_predictions > 0.5).float()
                correct_predictions_val += (val_predicted_labels == val_targets).sum().item()
                total_predictions_val += val_targets.numel()

        # Calculate average evaluation loss and accuracy
        avg_eval_loss = total_eval_loss / len(val_loader.dataset)
        eval_accuracy = correct_predictions_val / total_predictions_val

        if rank == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
            
            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Eval", avg_eval_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Eval", eval_accuracy, epoch)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        
        scheduler.step()
        #dist.barrier()

    # Save the model only on rank 0
    if rank == 0:
        writer.close()
        torch.save(ddp_model.state_dict(), "../trained/AnoMed/resnet_model_ddp.pth")

    cleanup_ddp()



def main():
    #world_size = torch.cuda.device_count()
    world_size = 2
    print(f"Using {world_size} GPUs")

    # Load dataset
    data_dir = "./data"
    celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)
    df = celeba.to_pandas()

    attr_names = df.columns[1:-1]
    image_col_name = df.columns[0]
    attributes = df[attr_names].values.astype(np.float32)
    inputs = df[image_col_name].values
    targets = (attributes + 1) / 2  # Convert -1,1 to 0,1

    # Split data
    train_percentage = 0.8
    val_percentage = 0.1
    test_percentage = 1 - train_percentage - val_percentage
    training_amount = int(train_percentage * len(inputs))
    validation_amount = int(val_percentage * len(inputs))

    train_inputs = inputs[:training_amount]
    val_inputs = inputs[training_amount:training_amount + validation_amount]
    train_targets = torch.tensor(targets[:training_amount], dtype=torch.float32)
    val_targets = torch.tensor(targets[training_amount:training_amount + validation_amount], dtype=torch.float32)

    # Use multiprocessing spawn for DDP
    mp.spawn(train_ddp,
             args=(world_size, train_inputs, train_targets, val_inputs, val_targets, 100, 8),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    main()

"""
Epoch [1/5], Train Loss: 0.3455, Train Accuracy: 0.8562
Eval Loss: 0.2460, Eval Accuracy: 0.8988
Epoch [2/5], Train Loss: 0.2225, Train Accuracy: 0.9076
Eval Loss: 0.2182, Eval Accuracy: 0.9067
Epoch [3/5], Train Loss: 0.2034, Train Accuracy: 0.9133
Eval Loss: 0.2068, Eval Accuracy: 0.9100
Epoch [4/5], Train Loss: 0.1941, Train Accuracy: 0.9166
Eval Loss: 0.2020, Eval Accuracy: 0.9112
Epoch [5/5], Train Loss: 0.1879, Train Accuracy: 0.9189
Eval Loss: 0.1988, Eval Accuracy: 0.9122
"""