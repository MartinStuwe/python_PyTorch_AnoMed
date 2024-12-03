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


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def train_ddp(rank, world_size, train_inputs, train_targets, eval_inputs, eval_targets, input_dim, num_epochs=100, batch_size=512):
    setup_ddp(rank, world_size)
    torch.manual_seed(0)  # Ensure deterministic behavior
    
    device = torch.device(f'cuda:{rank}')
    
    # Create model and move it to the appropriate device
    model = SimpleMLP(input_dim=input_dim).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # Create DataLoader with DistributedSampler for training
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Evaluation data (can use the entire dataset)
    eval_dataset = TensorDataset(eval_inputs, eval_targets)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Only rank 0 initializes the SummaryWriter
    if rank == 0:
        writer = SummaryWriter(log_dir="./tensorboard_logs/mlp")
        print(f"Amount of samples in dataset: {len(train_dataset)}")
        print(f"Amount of batches per epoch: {len(train_loader)}")

        # Training loop
    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch

        # Reset accumulators
        total_train_loss = 0
        correct_predictions_train = 0
        total_predictions_train = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs).view(-1, 1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

            # Calculate training accuracy for this batch
            predictions_train = torch.sigmoid(outputs)
            predicted_labels_train = (predictions_train > 0.5).float()
            correct_predictions_train += (predicted_labels_train == batch_targets).sum().item()
            total_predictions_train += batch_targets.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_predictions_train / total_predictions_train

        # Evaluation
        ddp_model.eval()
        total_eval_loss = 0
        correct_predictions_eval = 0
        total_predictions_eval = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in eval_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                
                outputs = ddp_model(batch_inputs).view(-1, 1)
                loss = criterion(outputs, batch_targets)
                total_eval_loss += loss.item()
                
                predictions_eval = torch.sigmoid(outputs)
                predicted_labels_eval = (predictions_eval > 0.5).float()
                correct_predictions_eval += (predicted_labels_eval == batch_targets).sum().item()
                total_predictions_eval += batch_targets.size(0)

        avg_eval_loss = total_eval_loss / len(eval_loader)
        eval_accuracy = correct_predictions_eval / total_predictions_eval

        if rank == 0:  # Only rank 0 logs and prints
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
            
            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Eval", avg_eval_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/Eval", eval_accuracy, epoch)
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        
        # Step the scheduler
        scheduler.step()

    # Save the model only on rank 0
    if rank == 0:
        writer.close()
        torch.save(ddp_model.state_dict(), "../trained/AnoMed/mlp_attractive.pth")

    cleanup_ddp()


        
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
    
    split_idx = int(0.8 * len(inputs))
    train_inputs, eval_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, eval_targets = targets[:split_idx], targets[split_idx:]
    
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    eval_inputs = torch.tensor(eval_inputs, dtype=torch.float32)
    eval_targets = torch.tensor(eval_targets, dtype=torch.float32)

    mp.spawn(train_ddp,
             args=(world_size, train_inputs, train_targets, eval_inputs, eval_targets, train_inputs.shape[1], 100, 8),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    mp.set_start_method('spawn')

    main()

