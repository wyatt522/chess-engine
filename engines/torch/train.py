import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import random_split, DataLoader # type: ignore
from torch.optim.lr_scheduler import MultiStepLR
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from auxiliary_func import check_memory




run_name = "turing_cluster_run3"


# Calcute memory distribution so that loading pgns is 10% of processed data, 1.5 gb leftover

total_mem = check_memory()
print(total_mem, flush=True)
# pgn_memory_mark = total_mem*30/31
# print(total_mem)
# print(pgn_memory_mark)

allocated_memory = 96
pgn_memory_mark = total_mem - allocated_memory/2
print(pgn_memory_mark, flush=True)


from auxiliary_func import load_dataset, encode_moves
#TODO: use zip instead, standardize
files = [file for file in os.listdir("../../data/Lichess_Elite_Database") if file.endswith(".pgn")]
# Sort by file size (ascending)
files_sorted = sorted(files, key=lambda f: os.path.getsize(os.path.join("../../data/Lichess_Elite_Database", f)))

LIMIT_OF_FILES = min(len(files_sorted), 80)

X, y, games_parsed, files_parsed = load_dataset(files_sorted, pgn_memory_mark=pgn_memory_mark, file_limit=LIMIT_OF_FILES)


X, y = np.array(X, dtype=np.float32), np.array(y)

y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)



X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


print("Completed Data Processing", flush=True)
print(f"GAMES PARSED: {games_parsed}", flush=True)
print(f"FILES PARSED: {files_parsed}", flush=True)
print(f"MOVES RECORDED: {len(y)}", flush=True)
available_gb = check_memory()
print(f"Available Memory: {available_gb}", flush=True)


from dataset import ChessDataset
from model import ChessModel


# Create Dataset
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Compute split sizes
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Then create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}', flush=True)

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

scheduler = MultiStepLR(optimizer, milestones=[60000, 150000, 300000], gamma=0.1)


# Get current time in a readable format
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a unique log directory
log_dir = f"../../runs/{run_name}_experiment_i{len(y)}_{current_time}"

# Create the SummaryWriter
writer = SummaryWriter(log_dir=log_dir)



num_epochs = 60
steps = 0
for epoch in range(num_epochs):
    start_time = time.time()

    # Training
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)  # Raw logits

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        steps += 1

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()


    # Stats
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60

    if epoch % 25 == 0:
        # Save the model
        torch.save(model.state_dict(), f"../../models/checkpoints/TORCH_{epoch}EPOCHS_{run_name}.pth")
    
    current_lr = scheduler.get_last_lr()[0]
    print(f'Steps: {steps}, Epoch: {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Time: {minutes}m{seconds}s, Learning Rate: {current_lr}', flush=True)

    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)

writer.close()



# Save the model
torch.save(model.state_dict(), f"../../models/{run_name}_final_model.pth", flush=True)



import pickle

with open(f"../../models/{run_name}_move_to_int", "wb") as file:
    pickle.dump(move_to_int, file)