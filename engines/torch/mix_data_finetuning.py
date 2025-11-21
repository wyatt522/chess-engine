import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import ChessDataset
from MiniMaia import MiniMaia, MiniMaiaSkip, MiniMaiaSkipFC
import pickle



run_name = "kai_minimaia_freeze"
move_to_int = "flipped_board_data"
original_dataset = "../../data/Lichess_Elite_Database/flipped_board_data_dataset.pth"
finetuning_dataset = "../../data/kai/kai_nakamura_dataset.pth"
allocated_memory = 60 # in GB Ram
num_epochs = 10
num_blocks = 6
reuse_model = "../../models/minimaia_with_skip_1024.pth"



X0, y0 = torch.load(original_dataset)
Xf, yf = torch.load(finetuning_dataset)

with open(f"../../models/{move_to_int}_move_to_int", "rb") as file:
    move_to_int = pickle.load(file)

num_classes = len(move_to_int)

print("Sucessfully loaded data")

# Create Finetuning Dataset
datasetf = ChessDataset(Xf, yf)

# Compute split sizes
train_size = int(0.9 * len(datasetf))
val_size = len(datasetf) - train_size

train_datasetf, val_datasetf = random_split(datasetf, [train_size, val_size])

# Then create DataLoaders
train_loaderf = DataLoader(train_datasetf, batch_size=128, shuffle=True)
val_loaderf = DataLoader(val_datasetf, batch_size=128, shuffle=False)

# Create Original Dataset
dataset0 = ChessDataset(X0, y0)

# Compute split sizes
train_size = int(0.9 * len(dataset0))
val_size = len(dataset0) - train_size

train_dataset0, val_dataset0 = random_split(dataset0, [train_size, val_size])

# Then create DataLoaders/Iterators
random_train0_sampler = RandomSampler(train_dataset0, replacement=True, num_samples=(num_epochs*(len(datasetf)//128)*1024))
train_loader0 = DataLoader(train_dataset0, batch_size=1024, sampler=random_train0_sampler)
train_iterator0 = iter(train_loader0)
val_loader0 = DataLoader(val_dataset0, batch_size=1024, shuffle=False)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}', flush=True)


model = MiniMaiaSkip(num_classes=num_classes, num_blocks=num_blocks, squeeze_layer=1024)
model.load_state_dict(torch.load(reuse_model, weights_only=True, map_location=device))

# Freeze everything except final fully connected layers
for name, child in model.named_children():
    for param in child.parameters():
        if name == 'fc1' or name == 'fc2':
            param.requires_grad = True
        else:
            param.requires_grad = False

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00002)

scheduler = MultiStepLR(optimizer, milestones=[500000], gamma=0.2)

# Get current time in a readable format
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a unique log directory
log_dir = f"../../runs/{run_name}_i{len(yf)}_{current_time}"

# Create the SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

steps = 0
for epoch in range(num_epochs):
    start_time = time.time()

    # Training
    model.train()
    running_loss = 0.0
    for inputsf, labelsf in tqdm(train_loaderf):


        inputsf, labelsf = inputsf.to(device), labelsf.to(device)  # Move data to GPU
        inputs0, labels0 = next(train_iterator0)
        inputs0, labels0 = inputs0.to(device), labels0.to(device)  # Move data to GPU

        labels_list = [labelsf, labels0]
        inputs_list = [inputsf, inputs0]

        finetuning_it = True
        for inputs, labels in zip(inputs_list, labels_list):
            optimizer.zero_grad()

            outputs = model(inputs)  # Raw logits

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            if finetuning_it:
                running_loss += loss.item()
                finetuning_it = False

            steps += 1


        

    # Validation
    model.eval()
    val_loss0 = 0.0
    val_lossf = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader0:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss0 += loss.item()

        for inputs, labels in val_loaderf:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_lossf += loss.item()



    # Stats
    avg_train_loss = running_loss / len(train_loaderf)
    avg_val_lossf = val_lossf / len(val_loaderf)

    end_time = time.time()
    epoch_time = end_time - start_time
    minutes: int = int(epoch_time // 60)
    seconds: int = int(epoch_time) - minutes * 60

    if epoch % 10 == 0:
        # Save the model
        torch.save(model.state_dict(), f"../../models/checkpoints/TORCH_{epoch}EPOCHS_{run_name}.pth")
    
    current_lr = scheduler.get_last_lr()[0]
    print(f'Steps: {steps}, Epoch: {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loaderf):.4f}, Validation Loss Original: {val_loss0 / len(val_loader0):.4f}, Validation Loss Finetuning: {val_lossf / len(val_loaderf):.4f}, Time: {minutes}m{seconds}s, Learning Rate: {current_lr}', flush=True)

    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    writer.add_scalar("Loss/validation", avg_val_lossf, epoch + 1)

writer.close()



# Save the model
torch.save(model.state_dict(), f"../../models/{run_name}_final_model.pth")