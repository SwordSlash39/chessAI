from agent import Agent
from gameChess import chess_game
import chess
import numpy as np

"""
--------------------------------------------------------------
NOT PART OF OFFICIAL MACHINE LEARNING PROGRAM. THIS PROGRAM IS TO FACILIATE A MODEL-VS-MODEL GAME WITHOUT TRAINING
THE GAME WILL BE UPLOADED INTO MODEL_GAME.TXT AND WILL BE IN PGN FORMAT
--------------------------------------------------------------
"""


def gamgin():
    agent = Agent()
    game = chess_game()
    fullgame = ""

    move = 1
    agent.model.eval()
    while True:
        """
        ----------------------------------------------------------------
        WHITE'S TURN
        ----------------------------------------------------------------
        """

        # get move
        final_move = agent.get_action(game.board, "white")
        mover = np.array(final_move, dtype="int16")
        fullgame += f"{move}. {game.board.san(list(game.board.legal_moves)[np.argmax(mover)])} "

        # perform move and get new state
        _, done_white = game.step(final_move)

        """
        ----------------------------------------------------------------
        BLACK'S TURN
        ----------------------------------------------------------------
        """
        if done_white:
            break
        # get move
        final_move = agent.get_action(game.board, "black")
        mover = np.array(final_move, dtype="int16")

        # perform move and get new state
        fullgame += f"{game.board.san(list(game.board.legal_moves)[np.argmax(mover)])} "
        _, done_black = game.step(final_move)

        if done_black:
            break
        print(f"Move: {move} completed!")
        move += 1
    with open("model_game.txt", "w") as f:
        f.write(fullgame)

if __name__ == '__main__':
    gamgin()
