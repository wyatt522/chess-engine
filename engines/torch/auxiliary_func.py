import numpy as np
from chess import Board, pgn
from tqdm import tqdm # type: ignore
import psutil
import random


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

def load_pgn(file_path, pgn_memory_mark = 1.0):
    games = []
    with open(file_path, 'r') as pgn_file:
        # print(file_path)
        i = 0
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            yield game

def create_input_for_nn(game, move_collection_prob = 0.33):
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

def load_dataset(files, pgn_memory_mark = 3.0, file_limit = 30):
    X, y = [], []
    games_parsed = files_parsed = 0

    for file in tqdm(files):

        for game in load_pgn(f"../../data/Lichess_Elite_Database/{file}"):
            games_parsed += 1
            x_temp, y_temp = create_input_for_nn(game)
            X.extend(x_temp)
            y.extend(y_temp)

            if games_parsed % 100 == 0:
                available_gb = check_memory()
                if available_gb < pgn_memory_mark:
                    print(f"Completed sampling {files_parsed} files with {available_gb} remaining")
                    return X, y

        files_parsed += 1
        if files_parsed >= file_limit:
            print(f"Completed sampling limit of files with {available_gb} remaining")
            return X, y