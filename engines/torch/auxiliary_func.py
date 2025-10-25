import numpy as np
from chess import Board
from tqdm import tqdm # type: ignore
import psutil
from collections import deque


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



def create_input_for_nn(games):

    LOW_MEMORY_GB = 5.0

    X = []
    y = []

    for i, game in enumerate(tqdm(games)):

        # Check memory every 100 games
        if i % 100 == 0:
            available_gb = check_memory()
            
            if available_gb < LOW_MEMORY_GB:
                print(f"Low memory hit at {i} games")
                print(available_gb)
                break

        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int
