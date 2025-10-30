import os
import numpy as np # type: ignore
import time
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import random_split, DataLoader # type: ignore
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from auxiliary_func import check_memory




run_name = "turing_cluster_run1"


# Calcute memory distribution so that loading pgns is 10% of processed data, 1.5 gb leftover

total_mem = check_memory()
pgn_memory_mark = total_mem*3/7
print(total_mem)
print(pgn_memory_mark)





from auxiliary_func import load_dataset, encode_moves
files = [file for file in os.listdir("../../data/Lichess_Elite_Database") if file.endswith(".pgn")]
# Sort by file size (ascending)
files_sorted = sorted(files, key=lambda f: os.path.getsize(os.path.join("../../data", f)))

LIMIT_OF_FILES = min(len(files_sorted), 80)
files_parsed = 0
games_parsed = 0
stop = False

X, y = load_dataset(files_sorted, pgn_memory_mark=pgn_memory_mark, file_limit=LIMIT_OF_FILES)


X, y = np.array(X, dtype=np.float32), np.array(y)

y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)



X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)



print(f"GAMES PARSED: {games_parsed}")
print(f"FILES PARSED: {files_parsed}")
print(f"MOVES RECORDED: {len(y)}")
available_gb = check_memory()
print(f"Available Memory: {available_gb}")


from dataset import ChessDataset
from model import ChessModel


# Create Dataset
dataset = ChessDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Compute split sizes
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Then create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Model Initialization
model = ChessModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



# Get current time in a readable format
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a unique log directory
log_dir = f"../../runs/{run_name}_experiment_i{len(y)}_{current_time}"

# Create the SummaryWriter
writer = SummaryWriter(log_dir=log_dir)



num_epochs = 500
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
        running_loss += loss.item()

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

    if epoch % 40 == 0:
        # Save the model
        torch.save(model.state_dict(), f"../../models/checkpoints/TORCH_{epoch}EPOCHS_{run_name}.pth")  
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Time: {minutes}m{seconds}s')

    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)

writer.close()



# Save the model
torch.save(model.state_dict(), f"../../models/{run_name}_final_model.pth")



import pickle

with open(f"../../models/{run_name}_move_to_int", "wb") as file:
    pickle.dump(move_to_int, file)