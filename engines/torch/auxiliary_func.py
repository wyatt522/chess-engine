import numpy as np
from chess import Board, pgn
from tqdm import tqdm # type: ignore
import psutil
import random
import os
import torch
import chess.gaviota


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((16, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 6 if board.turn != piece.color else 0
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    # From squares (14th board)
    # weighted map of your attacks (15th) and opponents(16th)
    piece_map = board.piece_map()
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        from_square = move.from_square
        row_to, col_to = divmod(to_square, 8)
        row_from, col_from = divmod(from_square, 8)
        matrix[12, row_to, col_to] = 1
        matrix[13, row_from, col_from] = 1

        if to_square in piece_map:
            matrix[14, row_to, col_to] += 0.0625
    
    board.turn = not board.turn

    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        if to_square in piece_map:
            matrix[15, row_to, col_to] += 0.0625

    board.turn = not board.turn

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

def create_input_for_nn(game, move_collection_prob = 0.15, endgame_select = True):
    X = []
    y = []

    board = game.board()
    move_num = 0
    for move in game.mainline_moves():
        move_num += 1
        if endgame_select:
            mult = 2 if move_num > 40 else 1
        if random.random() < move_collection_prob*mult:
            X.append(board_to_matrix(board))
            y.append(move.uci())

        board.push(move)
    return X, y

def create_input_for_nn_personal(game, user, move_collection_prob = 0.15, earlygame_select = True):
    headers = game.headers
    if headers["White"] == user:
        select_player = chess.WHITE
    elif headers["Black"] == user:
        select_player = chess.BLACK

    X = []
    y = []

    board = game.board()
    move_num = 0
    for move in game.mainline_moves():
        move_num += 1
        if board.turn == select_player:
            if earlygame_select:
                mult = 2 if move_num < 15 else 1
            if random.random() < move_collection_prob*mult:
                X.append(board_to_matrix(board))
                y.append(move.uci())

        board.push(move)
    return X, y

def create_input_for_nn_endgame_select(game):
    X = []
    y = []

    board = game.board()
    move_num = 0
    try:
        complete_game = "#" in game.end().san()

        if complete_game:
            for move in game.mainline_moves():
                move_num += 1
                if move_num > 50 and len(board.piece_map()) < 10:
                    X.append(board_to_matrix(board))
                    y.append(move.uci())

                board.push(move)
    except Exception as e:
        print(f"exception: " + e, flush=True)
    finally:
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
            x_temp, y_temp = create_input_for_nn_personal(game, "KaiNakamura", move_collection_prob=0.35, earlygame_select=True)
            X.extend(x_temp)
            y.extend(y_temp)

            if games_parsed % 500 == 0:
                available_gb = check_memory()
                if available_gb < pgn_memory_mark:
                    print(f"Completed sampling {files_parsed} files with {available_gb} remaining", flush=True)
                    return X, y, games_parsed, files_parsed

        files_parsed += 1
        if files_parsed >= LIMIT_OF_FILES:
            available_gb = check_memory()
            print(f"Completed sampling limit of files with {available_gb} remaining", flush=True)
            return X, y, games_parsed, files_parsed
        

def prepare_input(board: Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return X_tensor

# Function to make predictions
def probabilities_to_move(probabilities: np.ndarray, int_to_move: dict, board: Board, pseudo_temp: int = 2, endgame_safety: float = 0.75, tablebase_path: str = "/home/wyatt/Documents/capstone/Gaviota/gaviota"):
    
    argsort = np.argsort(probabilities)[::-1] # record what moves sorted probs correspond to
    sorted_probs = probabilities
    sorted_probs.sort()
    sorted_probs = sorted_probs[::-1]
    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]

    # # endgame process
    endgame_safety_select = random.random()
    if len(board.piece_map()) < 6 and endgame_safety_select < endgame_safety: 
        with chess.gaviota.open_tablebase(tablebase_path) as tablebase:
            curr_dtm = tablebase.probe_dtm(board)
            if curr_dtm > 0:
                for idx, prob in enumerate(sorted_probs):
                    move = int_to_move[argsort[idx]]
                    if idx > 20:
                        break
                    if move not in legal_moves_uci:
                        continue
                    
                    board.push_uci(move)
                    temp_dtm = tablebase.probe_dtm(board)

                    if board.is_checkmate():
                        board.pop()
                        return move
                    if 0 < -temp_dtm < curr_dtm:
                        board.pop()
                        return move
                    else:
                        board.pop()

    
    for i in range(10): # try finding a random legal move 10 times
        selection = random.random() ** pseudo_temp
        collective_sum = 0
        idx = 0
        for prob in sorted_probs:
            collective_sum += prob
            if selection < collective_sum:
                move = int_to_move[argsort[idx]]
                if move in legal_moves_uci:
                    return move
                else:
                    break
            idx += 1

    # selects most popular move
    idx = 0
    for prob in sorted_probs:
        move = int_to_move[argsort[idx]]
        if move in legal_moves_uci:
            return move
        idx += 1
    return None