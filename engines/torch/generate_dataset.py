import numpy as np # type: ignore
import torch
from auxiliary_func import check_memory, load_dataset, encode_moves
import pickle


dataset_name = "endgame_subdata"
move_to_int_name = "flipped_board_data"
data_folder = "../../data/Lichess_Elite_Database"
allocated_memory = 230 # in GB Ram

new_moves_to_int = False


# Calcute memory distribution so that 4/7 is dedicated to dataset pre tensor conversion, 3/7 saved for after


total_mem = check_memory() 
print(total_mem, flush=True)
pgn_memory_mark = total_mem - (4*allocated_memory)/7
print(pgn_memory_mark, flush=True)


np_X, np_y, games_parsed, files_parsed = load_dataset(data_folder=data_folder, pgn_memory_mark=pgn_memory_mark)


np_X, np_y = np.array(np_X, dtype=np.float32), np.array(np_y)
if new_moves_to_int:
    y, move_to_int = encode_moves(np_y)
    num_classes = len(move_to_int)
    with open(f"../../models/{dataset_name}_move_to_int", "wb") as file:
        pickle.dump(move_to_int, file)
else:
    with open(f"../../models/{move_to_int_name}_move_to_int", "rb") as file:
        move_to_int = pickle.load(file)
    
    X = []
    y = []
    for board, move in zip(np_X, np_y):
        if move in move_to_int:
            X.append(board)
            y.append(move_to_int[move])
        else:
            print("skipped move: " + move, flush=True)

    print(len(np_X))
    print(len(np_y))
    X, y = np.array(np_X, dtype=np.float32), np.array(y, dtype=np.float32)
    print(len(X))
    print(len(y))


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

torch.save((X, y), f"{data_folder}/{dataset_name}_dataset.pth")

print("Completed Data Processing", flush=True)
print(f"GAMES PARSED: {games_parsed}", flush=True)
print(f"FILES PARSED: {files_parsed}", flush=True)
print(f"MOVES RECORDED: {len(y)}", flush=True)
available_gb = check_memory()
print(f"Available Memory: {available_gb}", flush=True)
