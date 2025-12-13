import chess.engine
import chess
from tqdm import tqdm
import asyncio
import math


async def main() -> None:
    _, engine = await chess.engine.popen_uci("../../dist/MiniMaiaBot/MiniMaiaBot")
    _, stockfish = await chess.engine.popen_uci("/usr/local/bin/stockfish")
    
    await engine.configure({"ModelWeights": "default"})
    await stockfish.configure({"UCI_LimitStrength": True, "Skill Level": 20, "UCI_Elo": 1320})

    minimaia_w = 0
    minimaia_l = 0
    minimaia_t = 0

    for i in tqdm(range(100)):
        maia_c = chess.BLACK if i % 2 == 0 else chess.WHITE
        board = chess.Board()

        turn = chess.WHITE
        while board.outcome() == None:
            if turn == maia_c:
                move = await engine.play(board, limit=chess.engine.Limit(time=0.1))
            else:
                move = await stockfish.play(board, limit=chess.engine.Limit(time=0.25))
            
            board.push(move.move)

            turn = not turn
        
        if board.outcome().winner == None:
            minimaia_t += 1
        elif board.outcome().winner == maia_c:
            minimaia_w += 1
        elif board.outcome().winner != maia_c:
            minimaia_l += 1

    await engine.quit()
    await stockfish.quit()

    print(f"W: {minimaia_w} L: {minimaia_l} T: {minimaia_t}")
    wr = (minimaia_w + 0.5*minimaia_t)/(minimaia_w + minimaia_l + minimaia_t)

    elo = 1600 - 400*math.log10((1/wr) - 1)
    print(elo)

asyncio.run(main())