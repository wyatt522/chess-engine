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
from auxiliary_func import check_memory, load_dataset, encode_moves
from dataset import ChessDataset
from model import ChessModel
from model2 import ChessModel2
from model3 import ChessModel3
from model4 import ChessModel4
from model5 import ChessModel5
from model7 import ChessModel7
from model8 import ChessModel8
import pickle




run_name = "skip_connection_test"
dataset_name = "flipped_board_data"
data_folder = "../../data/Lichess_Elite_Database"
allocated_memory = 100 # in GB Ram
num_epochs = 70
dataset = "reuse"


# Calcute memory distribution so that 2/3 is dedicated to dataset pre tensor conversion, 1/2 saved for after

if dataset == "generate":

    total_mem = check_memory()
    print(total_mem, flush=True)
    pgn_memory_mark = total_mem - (2*allocated_memory)/3
    print(pgn_memory_mark, flush=True)


    X, y, games_parsed, files_parsed = load_dataset(data_folder=data_folder, pgn_memory_mark=pgn_memory_mark)


    X, y = np.array(X, dtype=np.float32), np.array(y)

    y, move_to_int = encode_moves(y)
    num_classes = len(move_to_int)


    with open(f"../../models/{dataset_name}_move_to_int", "wb") as file:
        pickle.dump(move_to_int, file)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    torch.save((X, y), f"{data_folder}/{dataset_name}_dataset.pth")

    print("Completed Data Processing", flush=True)
    print(f"GAMES PARSED: {games_parsed}", flush=True)
    print(f"FILES PARSED: {files_parsed}", flush=True)
    print(f"MOVES RECORDED: {len(y)}", flush=True)
    available_gb = check_memory()
    print(f"Available Memory: {available_gb}", flush=True)

elif dataset == "reuse":
    X, y = torch.load(f"{data_folder}/{dataset_name}_dataset.pth")

    with open(f"../../models/{dataset_name}_move_to_int", "rb") as file:
        move_to_int = pickle.load(file)

    num_classes = len(move_to_int)

    print("Sucessfully loaded data")




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
model = ChessModel8(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = MultiStepLR(optimizer, milestones=[50000, 250000, 400000], gamma=0.2)


# Get current time in a readable format
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a unique log directory
log_dir = f"../../runs/{run_name}_i{len(y)}_{current_time}"

# Create the SummaryWriter
writer = SummaryWriter(log_dir=log_dir)




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

    if epoch % 20 == 0:
        # Save the model
        torch.save(model.state_dict(), f"../../models/checkpoints/TORCH_{epoch}EPOCHS_{run_name}.pth")
    
    current_lr = scheduler.get_last_lr()[0]
    print(f'Steps: {steps}, Epoch: {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Time: {minutes}m{seconds}s, Learning Rate: {current_lr}', flush=True)

    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)

writer.close()



# Save the model
torch.save(model.state_dict(), f"../../models/{run_name}_final_model.pth")