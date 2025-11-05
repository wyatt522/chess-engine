import numpy as np
from chess import Board, pgn
from tqdm import tqdm # type: ignore
import psutil
import random
import os


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def check_memory():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    return available_gb

def load_pgn(file_path):
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game

def create_input_for_nn(game, move_collection_prob = 0.1):
    X = []
    y = []

    board = game.board()
    for move in game.mainline_moves():

        if random.random() < move_collection_prob:
            X.append(board_to_matrix(board))
            y.append(move.uci())

        board.push(move)
    return X, y


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

def load_dataset(data_folder, pgn_memory_mark = 3.0, file_limit = 80):
    files = [file for file in os.listdir(data_folder) if file.endswith(".pgn")]
    # Sort by file size (ascending)
    files_sorted = sorted(files, key=lambda f: os.path.getsize(os.path.join(data_folder, f)))

    LIMIT_OF_FILES = min(len(files_sorted), file_limit)

    X, y = [], []
    games_parsed = files_parsed = 0

    for file in tqdm(files_sorted):

        for game in load_pgn(f"{data_folder}/{file}"):
            games_parsed += 1
            x_temp, y_temp = create_input_for_nn(game)
            X.extend(x_temp)
            y.extend(y_temp)

            if games_parsed % 100 == 0:
                available_gb = check_memory()
                if available_gb < pgn_memory_mark:
                    print(f"Completed sampling {files_parsed} files with {available_gb} remaining", flush=True)
                    return X, y, games_parsed, files_parsed

        files_parsed += 1
        if files_parsed >= LIMIT_OF_FILES:
            available_gb = check_memory()
            print(f"Completed sampling limit of files with {available_gb} remaining", flush=True)
            return X, y, games_parsed, files_parsed