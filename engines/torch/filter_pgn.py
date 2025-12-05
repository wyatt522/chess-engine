import chess
import chess.pgn
from tqdm import tqdm
from auxiliary_func import load_pgn


data_folder = "../../data/monthly_lichess_data"
source_file = "november.pgn"
out_file = "LowEloFiltered.pgn"

with open(f"{data_folder}/{out_file}", "w", encoding="utf-8") as out_file:
    exporter = chess.pgn.FileExporter(out_file)
    for game in tqdm(load_pgn(f"{data_folder}/{source_file}")):
        headers = game.headers

        white_elo = int(headers.get("WhiteElo", "-1"))
        black_elo = int(headers.get("BlackElo", "-1"))
        
        if (0 < white_elo < 900)  and (0 < black_elo < 900):
            game.accept(exporter)
