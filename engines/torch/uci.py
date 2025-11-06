import sys
from auxiliary_func import board_to_matrix
import torch
from model import ChessModel
from model4 import ChessModel4
import pickle
import numpy as np
import chess
from chess import Board
import random

import os
import time
import torch
from torch.utils.data import DataLoader # type: ignore
from chess import pgn # type: ignore
from tqdm import tqdm # type: ignore



# -----------------------------
# Model + Prediction Functions
# -----------------------------

def prepare_input(board: Board):
    matrix = board_to_matrix(board)  # <-- You need your board_to_matrix implementation here
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor


# Detect if running from PyInstaller bundle
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS       # temp folder where PyInstaller unpacks files
else:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../../../")  # normal script location

MODEL_PATH = os.path.join(BASE_DIR, f"models/flipped_boards_model_final_model.pth")
MAPPING_PATH = os.path.join(BASE_DIR, f"models/flipped_board_data_move_to_int")

# Load mapping
with open(MAPPING_PATH, "rb") as file:
    move_to_int = pickle.load(file)
int_to_move = {v: k for k, v in move_to_int.items()}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ChessModel4(num_classes=len(move_to_int))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict_move(board: Board, pseudo_temp: int = 4):
    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    logits = logits.squeeze(0)  # Remove batch dimension
    
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    argsort = np.argsort(probabilities)[::-1] # record what moves sorted probs correspond to


    sorted_probs = probabilities
    sorted_probs.sort()
    sorted_probs = sorted_probs[::-1]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    # sorted_indices = np.argsort(probabilities)[::-1]
    for i in range(10): # try finding a legal move 10 times first
        selection = random.random() ** pseudo_temp
        collective_sum = 0
        idx = 0
        for prob in sorted_probs:
            # print(prob)
            collective_sum += prob
            if selection < collective_sum:
                move = int_to_move[argsort[idx]]
                if move in legal_moves_uci:
                    return move
                else:
                    break
            idx += 1
    return None

# -----------------------------
# UCI Protocol Loop
# -----------------------------

def uci_loop():
    board = Board()
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
            best_move = predict_move(board)
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
