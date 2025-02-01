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
import torch.nn.init as init  # Ensure this import is present

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageFile
import os
from io import BytesIO
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_ddp(rank, world_size):
    """
    Initializes the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # Ensure this port is free
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup_ddp():
    """
    Destroys the distributed environment.
    """
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
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
        else:
            # Otherwise, assume it's a file path and open it directly
            image = Image.open(image_data).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]
        return image, target

def initialize_weights_he(m):
    """
    Initialize the weights of Linear layers using He (Kaiming) Normal Initialization
    suitable for LeakyReLU activations.
    """
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            init.zeros_(m.bias)
        if m.out_features == 1:
            init.kaiming_normal_(m.weight)
        else:
            init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

# Function to train the model using DDP
def train_ddp(rank, world_size, train_inputs, train_targets, val_inputs, val_targets, num_epochs=20, batch_size=16):
    """
    Distributed training function.
    """
    # Initialize the process group for DDP
    setup_ddp(rank, world_size)

    # Set seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    import random
    random.seed(0)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{rank}')

    # Define transformations for the dataset
    custom_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Rank {rank}: Loading data...")

    # Prepare training dataset and sampler
    train_dataset = CelebADataset(train_inputs, train_targets, transform=custom_transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)

    # Prepare validation dataset and loader only for rank 0
    if rank == 0:
        val_dataset = CelebADataset(val_inputs, val_targets, transform=custom_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    else:
        val_loader = None

    print(f"Rank {rank}: Data loaded successfully.")

    # Load a pre-trained ResNet-18 model and modify the final layer
    resnet_model = models.resnet18(weights='DEFAULT')
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, 40)  # Adjust for 40 attributes

    # Initialize the new fully connected layer
    resnet_model.fc.apply(initialize_weights_he)

    # Move the model to the appropriate device and wrap with DDP
    resnet_model = resnet_model.to(device)
    ddp_model = DDP(resnet_model, device_ids=[rank])

    # Define loss function, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3, weight_decay=1e-2)  # Adjusted learning rate and weight decay
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # Ensure the save directory exists (only on rank 0)
    if rank == 0:
        os.makedirs('../trained/ResNet_models/', exist_ok=True)

    # Initialize SummaryWriter only on rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir="./tensorboard_logs/resnet/")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        print(f"Number of batches per epoch (training): {len(train_loader)}")
        print(f"Number of batches per epoch (validation): {len(val_loader)}")

    print(f"Rank {rank}: Starting training...")

    # Training loop
    for epoch in range(num_epochs):
        dist.barrier()
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        #print("I Shuffled data")

        # Reset training accumulators
        total_train_loss = 0.0
        correct_predictions_train = 0
        total_predictions_train = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            #print("I train on a batch")
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)  # Optional: Gradient clipping
            optimizer.step()

            # Accumulate training loss and accuracy
            total_train_loss += loss.item() * images.size(0)  # Multiply by batch size
            predictions = torch.sigmoid(outputs)
            predicted_labels = (predictions > 0.5).float()
            correct_predictions_train += (predicted_labels == targets).sum().item()
            total_predictions_train += targets.numel()

        # Aggregate training metrics across all processes
        train_loss_tensor = torch.tensor([total_train_loss], dtype=torch.float32, device=device)
        correct_predictions_train_tensor = torch.tensor([correct_predictions_train], dtype=torch.float32, device=device)
        total_predictions_train_tensor = torch.tensor([total_predictions_train], dtype=torch.float32, device=device)

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_predictions_train_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_predictions_train_tensor, op=dist.ReduceOp.SUM)

        global_total_train_loss = train_loss_tensor.item()
        global_correct_predictions_train = correct_predictions_train_tensor.item()
        global_total_predictions_train = total_predictions_train_tensor.item()

        # Compute global averaged metrics
        avg_train_loss = global_total_train_loss / len(train_loader.dataset)
        train_accuracy = global_correct_predictions_train / global_total_predictions_train

        # Validation phase handled only by rank 0
        if rank == 0:
            ddp_model.eval()

            # Reset validation accumulators
            total_eval_loss = 0.0
            correct_predictions_val = 0
            total_predictions_val = 0

            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images, val_targets = val_images.to(device), val_targets.to(device)
                    val_outputs = ddp_model(val_images)
                    val_loss = criterion(val_outputs, val_targets)
                    total_eval_loss += val_loss.item() * val_images.size(0)  # Multiply by batch size

                    # Calculate validation accuracy
                    val_predictions = torch.sigmoid(val_outputs)
                    val_predicted_labels = (val_predictions > 0.5).float()
                    correct_predictions_val += (val_predicted_labels == val_targets).sum().item()
                    total_predictions_val += val_targets.numel()

            # Compute average validation loss and accuracy
            avg_val_loss = total_eval_loss / len(val_loader.dataset)
            eval_accuracy = correct_predictions_val / total_predictions_val

            # Logging and printing
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Eval Acc: {eval_accuracy:.4f}")

            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Eval", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Eval", eval_accuracy, epoch)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

            # Save the model checkpoint after each epoch
            model_save_path = f'../trained/ResNet_models/resnet_model_epoch_{epoch+1}.pth'
            torch.save(ddp_model.module.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        #elif rank == 0:
            # If there's no validation loader, still log training metrics
        #    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        #    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        #    writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        #    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # Scheduler step
        scheduler.step()
        dist.barrier()

    # Save the final model after all epochs
    if rank == 0:
        writer.close()
        # Save the final model after all epochs
        final_model_save_path = '../trained/ResNet_models/resnet_model_final.pth'
        torch.save(ddp_model.module.state_dict(), final_model_save_path)
        print(f"Final model saved to {final_model_save_path}")

    # Cleanup the DDP process group
    cleanup_ddp()

# NOTE: Uses test set, avoid peeking at results until end!
def evaluate_model(test_inputs, test_targets, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = models.resnet18(weights='DEFAULT')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 40)  # Adjust for 40 attributes
    
    # Initialize the new fully connected layer
    def initialize_weights_he(m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                init.zeros_(m.bias)
            if m.out_features == 1:
                init.xavier_normal_(m.weight)
            else:
                init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        
    model = model.to(device)
    model.load_state_dict(torch.load('../trained/ResNet_models/resnet_model_final.pth'))
    model.eval()
    
    # Define transformations for the dataset
    custom_transform = transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create DataLoader for test set
    test_dataset = CelebADataset(test_inputs, test_targets, transform=custom_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=False)
    
    criterion = nn.BCEWithLogitsLoss()
    
    total_test_loss = 0.0
    correct_predictions_test = 0
    total_predictions_test = 0

    with torch.no_grad():
        for test_images, test_targets_batch in test_loader:
            test_images, test_targets_batch = test_images.to(device), test_targets_batch.to(device)
            test_outputs = model(test_images)
            test_loss = criterion(test_outputs, test_targets_batch)
            total_test_loss += test_loss.item() * test_images.size(0)  # Multiply by batch size

            # Calculate test accuracy
            test_predictions = torch.sigmoid(test_outputs)
            test_predicted_labels = (test_predictions > 0.5).float()
            correct_predictions_test += (test_predicted_labels == test_targets_batch).sum().item()
            total_predictions_test += test_targets_batch.numel()

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    test_accuracy = correct_predictions_test / total_predictions_test

    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

def main():
    #world_size = torch.cuda.device_count()
    world_size = 1
    if world_size < 2:
        print(f"Warning: Detected {world_size} GPU(s). The script is configured for 2 GPUs.")
        world_size = min(world_size, 2)  # Adjust world_size if fewer GPUs are available
    print(f"Using {world_size} GPU(s) for training.")
    
    # Load dataset
    data_dir = "./data"
    celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)
    df = celeba.to_pandas()

    attr_names = df.columns[1:-1]
    image_col_name = df.columns[0]
    attributes = df[attr_names].values.astype(np.float32)
    inputs = df[image_col_name].values
    targets = (attributes + 1) / 2  # Convert -1,1 to 0,1

    # First, split into training and temporary sets (80% train, 20% temp)
    train_inputs, temp_inputs, train_targets_np, temp_targets_np = train_test_split(
        inputs, targets, test_size=0.2, random_state=42, shuffle=True
    )

    # Then, split the temporary set into validation and test sets (10% each)
    val_inputs, test_inputs, val_targets_np, test_targets_np = train_test_split(
        temp_inputs, temp_targets_np, test_size=0.5, random_state=42, shuffle=True
    )

    # After splitting
    train_targets = torch.tensor(train_targets_np, dtype=torch.float32)
    val_targets = torch.tensor(val_targets_np, dtype=torch.float32)
    test_targets = torch.tensor(test_targets_np, dtype=torch.float32)

    # Use multiprocessing spawn for DDP
    mp.spawn(train_ddp,
             args=(world_size, train_inputs, train_targets, val_inputs, val_targets, 20, 8),  # 20 epochs, batch size 8
             nprocs=world_size,
             join=True)

    # NOTE: AT END: Evaluate on the test set after training
    # Uncomment the following line when you're ready to evaluate
    # evaluate_model(test_inputs, test_targets, input_dim=None)  # input_dim is irrelevant for ResNet

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
