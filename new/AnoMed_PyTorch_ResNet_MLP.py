import os
import time
from PIL import Image
from io import BytesIO


import numpy as np

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset, DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from torchvision import transforms, models

from datasets import load_dataset


torch.manual_seed(0)


# Define custom transform
custom_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class CelebADataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images  # List of image paths or byte data dictionaries
        self.targets = targets  # Corresponding targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_data = self.images[idx]

        if isinstance(image_data, dict):  # If image data is stored as a dict (e.g., {'bytes': ...})
            img_bytes = image_data['bytes']
            image = Image.open(BytesIO(img_bytes))
        else:  # Otherwise, assume it's a file path
            image = Image.open(image_data)

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]
        return image, target

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=40):
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.LeakyReLU()

        self.fc2_proj = nn.Linear(64, 128)  # Projection to match dimensions for skip connection

        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU()

        self.fc3_proj = nn.Linear(128, 256)  # Projection to match dimensions for skip connection

        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.LeakyReLU()

        self.fc5 = nn.Linear(512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu5 = nn.LeakyReLU()

        self.fc5_proj = nn.Linear(512, 1024)  # Projection to match dimensions for skip connection

        self.fc6 = nn.Linear(1024, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.LeakyReLU()

        self.fc7 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.relu7 = nn.LeakyReLU()

        self.fc8 = nn.Linear(1024, 128)
        self.bn8 = nn.BatchNorm1d(128)
        self.relu8 = nn.LeakyReLU()

        self.fc9 = nn.Linear(128, 64)
        self.bn9 = nn.BatchNorm1d(64)
        self.relu9 = nn.LeakyReLU()

        self.fc10 = nn.Linear(64, 32)
        self.bn10 = nn.BatchNorm1d(32)
        self.relu10 = nn.LeakyReLU()

        self.fc11 = nn.Linear(32, 16)
        self.bn11 = nn.BatchNorm1d(16)
        self.relu11 = nn.LeakyReLU()

        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        # First layer block
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        # Second layer block
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        # Skip connection from x1 to x2, with projection to match dimensions
        x2 = x2 + self.fc2_proj(x1)

        # Third layer block
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        # Skip connection from x2 to x3, with projection
        x3 = x3 + self.fc3_proj(x2)

        # Fourth and fifth layer blocks
        x4 = self.fc4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        x5 = self.fc5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)

        # Skip connection from x4 to x5, with projection
        x5 = x5 + self.fc5_proj(x4)

        # Continue without skip connections from here
        x6 = self.fc6(x5)
        x6 = self.bn6(x6)
        x6 = self.relu6(x6)

        x7 = self.fc7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)

        x8 = self.fc8(x7)
        x8 = self.bn8(x8)
        x8 = self.relu8(x8)

        x9 = self.fc9(x8)
        x9 = self.bn9(x9)
        x9 = self.relu9(x9)

        x10 = self.fc10(x9)
        x10 = self.bn10(x10)
        x10 = self.relu10(x10)

        x11 = self.fc11(x10)
        x11 = self.bn11(x11)
        x11 = self.relu11(x11)

        output = self.fc_out(x11)

        return output

class CombinedModel(nn.Module):
    def __init__(self, resnet, mlp):
        super(CombinedModel, self).__init__()
        self.resnet = resnet
        self.mlp = mlp

    def forward(self, x):
        resnet_output = self.resnet(x)
        output = self.mlp(resnet_output)
        return output

def train_ddp(rank, world_size, train_inputs, train_targets, val_inputs, val_targets, num_epochs=100, batch_size=1024):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{rank}')

        # Load pre-trained ResNet model and freeze its parameters
        resnet_model = models.resnet18(pretrained=True)


        # Modify the final layer of ResNet to output 40 attributes
        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_features, 40)
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model = resnet_model.to(device)

        # Convert ResNet model to DDP
        #NOTE: has no required gradients, but not usable for this, got an error 
        # i.e. RuntimeError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient. 
        #resnet_model = DDP(resnet_model, device_ids=[rank])

        # Load pre-trained ResNet state dict
        checkpoint = torch.load("../trained/AnoMed/resnet_model.pth", weights_only=True)
        # Remove 'module.' from the keys in state_dict
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        resnet_model.load_state_dict(new_state_dict)

        # Create SimpleMLP model and convert to DDP
        mlp = SimpleMLP().to(device)
        mlp = DDP(mlp, device_ids=[rank])

        # Load pre-trained MLP state dict
        mlp_checkpoint = torch.load("../trained/AnoMed/mlp_attractive.pth", weights_only=True)
        mlp.load_state_dict(mlp_checkpoint)

        # Combine ResNet and MLP into a single model
        combined_model = CombinedModel(resnet_model, mlp).to(device)
        combined_model = DDP(combined_model, device_ids=[rank])

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-1)
        scheduler = ExponentialLR(optimizer, gamma=0.95)

        # Create Dataset and DataLoader with DistributedSampler
        train_dataset = CelebADataset(train_inputs, train_targets, transform=custom_transform)
        val_dataset = CelebADataset(val_inputs, val_targets, transform=custom_transform)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=3)
        # 2-4*#GPU
            
        # Initialize the TensorBoard writer only for rank 0
        if rank == 0:
            writer = SummaryWriter(log_dir="./tensorboard_logs")
    # Training loop
    for epoch in range(num_epochs):
        combined_model.train()
        train_sampler.set_epoch(epoch)  # Set sampler for each epoch

        # Reset accumulators
        total_train_loss = 0
        total_val_loss = 0
        correct_predictions_train = 0
        total_predictions_train = 0
        correct_predictions_val = 0
        total_predictions_val = 0

        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_iter):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()

            outputs = combined_model(batch_inputs).view(-1, 1)

            loss = criterion(outputs, batch_targets)
            loss.backward()

            optimizer.step()
            total_train_loss += loss.item()

            # Calculate train accuracy for this batch
            predictions_train = torch.sigmoid(outputs)
            predicted_labels_train = (predictions_train > 0.5).float()
            correct_predictions_train += (predicted_labels_train == batch_targets).sum().item()
            total_predictions_train += batch_targets.size(0)

            # Evaluation step after each training batch
            combined_model.eval()  # Switch to eval mode

            with torch.no_grad():
                try:
                    val_inputs, val_targets = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)  # Reset if exhausted
                    val_inputs, val_targets = next(val_iter)

                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = combined_model(val_inputs).view(-1, 1)
                val_loss = criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()

                # Calculate validation accuracy for this batch
                predictions_val = torch.sigmoid(val_outputs)
                predicted_labels_val = (predictions_val > 0.5).float()
                correct_predictions_val += (predicted_labels_val == val_targets).sum().item()
                total_predictions_val += val_targets.size(0)

            combined_model.train()  # Switch back to training mode

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            train_accuracy = correct_predictions_train / total_predictions_train
            val_accuracy = correct_predictions_val / total_predictions_val
            print(f"Train Acc: {train_accuracy:.4f}")
            print(f"Eval Acc: {val_accuracy:.4f}")

            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        scheduler.step()

    if rank == 0:
        writer.close()
        torch.save(combined_model.state_dict(), '../trained/combined_model.pth')

cleanup_ddp()




def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    data_dir = "../data"
    celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)
    
    df = celeba.to_pandas()
    attr_names = df.columns[1:-1]
    image_col_name = df.columns[0]
    
    inputs = df[image_col_name].values
    attributes = df[attr_names].values.astype(np.float32)
    targets = (attributes + 1) / 2  # Normalize to 0,1
    
    attr_to_predict = 'Attractive'
    attr_idx = list(attr_names).index(attr_to_predict)
    
    train_inputs = inputs[:int(0.8 * len(inputs))]
    train_targets = torch.tensor(targets[:int(0.8 * len(inputs)), attr_idx].reshape(-1, 1), dtype=torch.float32)
    val_inputs = inputs[int(0.8 * len(inputs)):]
    val_targets = torch.tensor(targets[int(0.8 * len(inputs)):, attr_idx].reshape(-1, 1), dtype=torch.float32)


    mp.spawn(train_ddp,
             args=(world_size, train_inputs, train_targets, val_inputs, val_targets),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    main()

"""
Batch [140], Train Accuracy: 0.8372, Eval Accuracy: 0.8243
Batch [141], Train Accuracy: 0.8372, Eval Accuracy: 0.8243
Batch [142], Train Accuracy: 0.8371, Eval Accuracy: 0.8243
Batch [143], Train Accuracy: 0.8371, Eval Accuracy: 0.8244
Batch [144], Train Accuracy: 0.8372, Eval Accuracy: 0.8245
Batch [145], Train Accuracy: 0.8371, Eval Accuracy: 0.8243
Batch [146], Train Accuracy: 0.8370, Eval Accuracy: 0.8242
Batch [147], Train Accuracy: 0.8370, Eval Accuracy: 0.8245
Batch [148], Train Accuracy: 0.8369, Eval Accuracy: 0.8245
Batch [149], Train Accuracy: 0.8369, Eval Accuracy: 0.8244
Batch [150], Train Accuracy: 0.8369, Eval Accuracy: 0.8246
Batch [151], Train Accuracy: 0.8367, Eval Accuracy: 0.8246
Batch [152], Train Accuracy: 0.8369, Eval Accuracy: 0.8245
Batch [153], Train Accuracy: 0.8370, Eval Accuracy: 0.8243
Batch [154], Train Accuracy: 0.8370, Eval Accuracy: 0.8245
Batch [155], Train Accuracy: 0.8370, Eval Accuracy: 0.8246
Batch [156], Train Accuracy: 0.8369, Eval Accuracy: 0.8248
Batch [157], Train Accuracy: 0.8366, Eval Accuracy: 0.8246
Batch [158], Train Accuracy: 0.8366, Eval Accuracy: 0.8245
Batch [159], Train Accuracy: 0.8366, Eval Accuracy: 0.8246
Epoch [5/5], Train Loss: 0.3405, Val Loss: 1.4406
"""
# NOTE: ACC fuer alle Daten

#TODO: PDF verfassen
