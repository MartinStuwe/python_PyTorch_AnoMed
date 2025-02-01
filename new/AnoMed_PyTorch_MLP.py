import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset, DistributedSampler
from sklearn.model_selection import train_test_split

import torch.nn.init as init

def initialize_weights_he(m):
    """
    Initialize the weights of Linear layers using He (Kaiming) Normal Initialization
    suitable for LeakyReLU activations.
    """
    if isinstance(m, nn.Linear):
        # He (Kaiming) Normal Initialization for weights
        if m.bias is not None:
            init.zeros_(m.bias)
        if m.out_features == 1:
            init.xavier_normal_(m.weight)
        else:
            init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')


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
        # TODO: nn.init.kaiming_uniform(x1.weight)
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train_ddp(rank, world_size, train_inputs, train_targets, val_inputs, val_targets, input_dim, num_epochs=100, batch_size=16):
    setup_ddp(rank, world_size)
    torch.manual_seed(0)  # Ensure deterministic behavior
    np.random.seed(0)
    import random
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(f'cuda:{rank}')

    # Ensure the save directory exists (only on rank 0)
    if rank == 0:
        os.makedirs('../trained/MLP_models/', exist_ok=True)  # Changed directory to 'MLP_models'

    # Create model and move it to the appropriate device
    model = SimpleMLP(input_dim=input_dim).to(device)
    model.apply(initialize_weights_he)

    ddp_model = DDP(model, device_ids=[rank])

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.95)  # Revert to ExponentialLR with desired gamma

    # Create DataLoader with DistributedSampler for training
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)

    # Validation data (handled by rank 0)
    if rank == 0:
        val_dataset = TensorDataset(val_inputs, val_targets)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    else:
        val_loader = None

    # Only rank 0 initializes the SummaryWriter
    if rank == 0:
        writer = SummaryWriter(log_dir="./tensorboard_logs/mlp")
        print(f"Amount of samples in training dataset: {len(train_dataset)}")
        if val_loader:
            print(f"Amount of samples in validation dataset: {len(val_dataset)}")
            print(f"Amount of batches per epoch (training): {len(train_loader)}")
            print(f"Amount of batches per epoch (validation): {len(val_loader)}")

    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch
        
        # Reset training accumulators
        total_train_loss = 0.0
        correct_predictions_train = 0
        total_predictions_train = 0

        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device).view(-1, 1).float()
            
            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs).view(-1, 1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)  # Optional: Gradient clipping
            optimizer.step()
            
            total_train_loss += loss.item() * batch_inputs.size(0)  # Accumulate loss
            
            # Calculate train accuracy for this batch
            predictions_train = torch.sigmoid(outputs)
            predicted_labels_train = (predictions_train > 0.5).float()
            correct_predictions_train += (predicted_labels_train == batch_targets).sum().item()
            total_predictions_train += batch_targets.size(0)

        # Compute local training loss and accuracy
        avg_train_loss = total_train_loss / len(train_sampler)
        train_accuracy = correct_predictions_train / total_predictions_train

        # Aggregate training metrics across all processes
        train_loss_tensor = torch.tensor(avg_train_loss, device=device)
        train_accuracy_tensor = torch.tensor(train_accuracy, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_accuracy_tensor, op=dist.ReduceOp.SUM)

        global_avg_train_loss = train_loss_tensor.item() / world_size
        global_train_accuracy = train_accuracy_tensor.item() / world_size

        # Validation (only rank 0)
        if rank == 0 and val_loader is not None:
            ddp_model.eval()
            total_val_loss = 0.0
            correct_predictions_val = 0
            total_predictions_val = 0

            with torch.no_grad():
                for val_inputs_batch, val_targets_batch in val_loader:
                    val_inputs_batch, val_targets_batch = val_inputs_batch.to(device), val_targets_batch.to(device).view(-1,1).float()
                    val_outputs = ddp_model(val_inputs_batch).view(-1, 1)
                    val_loss = criterion(val_outputs, val_targets_batch)
                    total_val_loss += val_loss.item() * val_inputs_batch.size(0)

                    # Calculate validation accuracy
                    predictions_val = torch.sigmoid(val_outputs)
                    predicted_labels_val = (predictions_val > 0.5).float()
                    correct_predictions_val += (predicted_labels_val == val_targets_batch).sum().item()
                    total_predictions_val += val_targets_batch.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            val_accuracy = correct_predictions_val / total_predictions_val

            # Logging
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {global_avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Train Acc: {global_train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

            writer.add_scalar('Loss/Train', global_avg_train_loss, epoch)
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Train', global_train_accuracy, epoch)
            writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

            # Save the model checkpoint after each epoch
            model_save_path = f'../trained/MLP_models/MLP_model_epoch_{epoch+1}.pth'
            torch.save(ddp_model.module.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        # Scheduler step
        scheduler.step()

    if rank == 0:
        writer.close()
        # Save the final model after all epochs
        final_model_save_path = '../trained/MLP_models/MLP_model_final.pth'
        torch.save(ddp_model.module.state_dict(), final_model_save_path)
        print(f"Final model saved to {final_model_save_path}")

    cleanup_ddp()

# NOTE: Uses test set, avoid peeking at results until end!
def evaluate_model(test_inputs, test_targets, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = SimpleMLP(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load('../trained/MLP_models/MLP_model_final.pth'))
    model.eval()
    
    # Create DataLoader for test set
    test_dataset = TensorDataset(test_inputs, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = nn.BCEWithLogitsLoss()
    
    total_test_loss = 0.0
    correct_predictions_test = 0
    total_predictions_test = 0

    with torch.no_grad():
        for test_inputs_batch, test_targets_batch in test_loader:
            test_inputs_batch, test_targets_batch = test_inputs_batch.to(device), test_targets_batch.to(device).view(-1,1).float()
            test_outputs = model(test_inputs_batch).view(-1, 1)
            test_loss = criterion(test_outputs, test_targets_batch)
            total_test_loss += test_loss.item() * test_inputs_batch.size(0)

            # Calculate test accuracy
            predictions_test = torch.sigmoid(test_outputs)
            predicted_labels_test = (predictions_test > 0.5).float()
            correct_predictions_test += (predicted_labels_test == test_targets_batch).sum().item()
            total_predictions_test += test_targets_batch.size(0)

    avg_test_loss = total_test_loss / len(test_loader.dataset)
    test_accuracy = correct_predictions_test / total_predictions_test

    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    # Load data
    data_dir = "./data"
    celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)
    df = celeba.to_pandas()
    
    attribute_names = df.columns[1:-1]
    attributes = df[attribute_names].values.astype(np.float32)
    
    attr_to_predict = 'Attractive'
    attr_idx = list(attribute_names).index(attr_to_predict)
    inputs = attributes.copy()
    inputs[:, attr_idx] = 0
    targets = attributes[:, attr_idx:attr_idx + 1]
    targets = (targets + 1) / 2  # Convert -1,1 to 0,1
    
    # First, split into training and temporary sets (e.g., 80% train, 20% temp)
    train_inputs, temp_inputs, train_targets_np, temp_targets_np = train_test_split(
        inputs, targets, test_size=0.2, random_state=42, shuffle=True
    )

    # Then, split the temporary set into validation and test sets (e.g., 10% each)
    val_inputs, test_inputs, val_targets_np, test_targets_np = train_test_split(
        temp_inputs, temp_targets_np, test_size=0.5, random_state=42, shuffle=True
    )
        
    train_targets = torch.tensor(train_targets_np, dtype=torch.float32)
    val_targets = torch.tensor(val_targets_np, dtype=torch.float32)
    test_targets = torch.tensor(test_targets_np, dtype=torch.float32)

    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32)

    # Train the model
    mp.spawn(train_ddp,
             args=(world_size, train_inputs, train_targets, val_inputs, val_targets, train_inputs.shape[1], 100, 16),
             nprocs=world_size,
             join=True)
    
    # NOTE: AT END: Evaluate on the test set after training
    #evaluate_model(test_inputs, test_targets, input_dim=train_inputs.shape[1])

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
