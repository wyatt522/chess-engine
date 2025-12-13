import sys
from auxiliary_func import prepare_input, probabilities_to_move
import torch
from MiniMaia import MiniMaiaSkip
import pickle
import numpy as np
from chess import Board

import os
import time
import torch
import yaml


# Detect if running from PyInstaller bundle
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS       # temp folder where PyInstaller unpacks files
else:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../../")  # normal script location

with open(os.path.join(BASE_DIR, "uci_config.yaml")) as file:
    config = yaml.safe_load(file)

MAPPING_PATH = config['MoveToIntPath']
TABLEBASE_PATH = config['GaviotaPath']

ModelWeights = config["ModelWeights"]

move_selection_options = {}
move_selection_options["PsuedoTemp"] = config.get("PseudoTemp", 5)
move_selection_options["StartGameTemp"] = config.get("StartGameTemp", 1.5)
move_selection_options["EarlyGameTemp"] = config.get("EarlyGameTemp", 3)

move_selection_options["EndgameCorrection"] = config.get("EndgameCorrection", 0.9)

# Load mapping
with open(MAPPING_PATH, "rb") as file:
    move_to_int = pickle.load(file)
int_to_move = {v: k for k, v in move_to_int.items()}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MiniMaiaSkip(num_classes=len(move_to_int))
model.load_state_dict(torch.load(ModelWeights["default"], map_location=device))
model.to(device)
model.eval()

# -----------------------------
# UCI Protocol Loop
# -----------------------------

def uci_loop():
    board = Board()
    model_weights_name = "default"
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        if line == "uci":
            print("id name MiniMaiaBot")
            print("id author Wyatt")
            print("option name StartGameTemp type spin default 1500 min 500 max 10000")
            print("option name EarlyGameTemp type spin default 3000 min 500 max 10000")
            print("option name PsuedoTemp type spin default 5000 min 500 max 10000")
            print("option name EndgameCorrection type spin default 900 min 1 max 1000")
            print("option name ModelWeights type string default default")
            
            print("uciok")


            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line == "printboard":
            print(board)
        
        elif line.startswith("setoption name"):
            parts = line.split(" ")
            if "name" in parts and "value" in parts:
                name_idx = parts.index("name") + 1
                val_idx = parts.index("value") + 1
                if parts[name_idx] == "ModelWeights":
                    if parts[val_idx] in ModelWeights and parts[val_idx] != model_weights_name:
                        model_weights_name = parts[val_idx]
                        model.load_state_dict(torch.load(ModelWeights[model_weights_name], map_location=device))
                    else:
                        pass
                else:
                    move_selection_options[parts[name_idx]] = int(parts[val_idx])/1000


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

            if len(board.piece_map()) == 32:
                temp = move_selection_options["StartGameTemp"]
            elif len(board.piece_map()) > 29:
                temp = move_selection_options["EarlyGameTemp"] 
            else:
                temp = move_selection_options["PsuedoTemp"]

            best_move = probabilities_to_move(probabilities=probabilities, int_to_move=int_to_move, 
                                                board=board, pseudo_temp=temp, endgame_safety=move_selection_options["EndgameCorrection"], 
                                                tablebase_path=TABLEBASE_PATH)

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
