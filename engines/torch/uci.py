import sys
from auxiliary_func import prepare_input, probabilities_to_move
import torch
from MiniMaia import MiniMaiaSkipFC
import pickle
import numpy as np
from chess import Board

import os
import time
import torch


# Detect if running from PyInstaller bundle
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS       # temp folder where PyInstaller unpacks files
    TABLEBASE_PATH = os.path.join(os.path.dirname(__file__), BASE_DIR, "/gaviota")
else:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../../")  # normal script location
    TABLEBASE_PATH = os.path.join(os.path.dirname(__file__), BASE_DIR, "../Gaviota/gaviota")

MODEL_PATH = os.path.join(BASE_DIR, f"models/minimaia_with_skip_fc.pth")
MAPPING_PATH = os.path.join(BASE_DIR, f"models/flipped_board_data_move_to_int")
# Load mapping
with open(MAPPING_PATH, "rb") as file:
    move_to_int = pickle.load(file)
int_to_move = {v: k for k, v in move_to_int.items()}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MiniMaiaSkipFC(num_classes=len(move_to_int))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# UCI Protocol Loop
# -----------------------------

def uci_loop():
    board = Board()
    endgame_boost = 0.8
    psuedo_temp = 2
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        if line == "uci":
            print("id name TorchEngine")
            print("id author Wyatt")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()
        
        # elif line.startswith("setoption name"):
        #     parts = line.split(" ")
        #     if "temperature" in parts:
        #         temp_idx = parts.index("temperature") + 1
        #         if 0 < parts[temp_idx]:
        #             psuedo_temp = parts[temp_idx]

        #     if ""


        elif line.startswith("position"):
            parts = line.split(" ")
            if "startpos" in parts:
                board.set_fen(Board().fen())
                moves_index = parts.index("moves") + 1 if "moves" in parts else None
                if moves_index:
                    for move in parts[moves_index:]:
                        board.push_uci(move)
            elif "fen" in parts:
                fen_index = parts.index("fen") + 1
                fen = " ".join(parts[fen_index:fen_index+6])
                board.set_fen(fen)
                if "moves" in parts:
                    moves_index = parts.index("moves") + 1
                    for move in parts[moves_index:]:
                        board.push_uci(move)

        elif line.startswith("go"):
            X_tensor = prepare_input(board).to(device)
    
            with torch.no_grad():
                logits = model(X_tensor)
            
            logits = logits.squeeze(0)  # Remove batch dimension
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities

            best_move = probabilities_to_move(probabilities=probabilities, int_to_move=int_to_move, board=board, tablebase_path=TABLEBASE_PATH)

            if best_move:
                print(f"bestmove {best_move}")
                sys.stdout.flush()
            else:
                print("bestmove 0000")
                sys.stdout.flush()

        elif line == "quit":
            break

if __name__ == "__main__":
    uci_loop()
