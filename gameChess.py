import chess
import json
import numpy as np


class chess_game:
    def __init__(self, FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.board = chess.Board()
        self.color_pieces = {"white": ['q', 'k', 'r', 'b', 'n', 'p'], "black": ['Q', 'K', 'R', 'B', 'N', 'P']}
        self.n_games = 0
        self.pieces = ['q', 'k', 'r', 'b', 'n', 'p', 'Q', 'K', 'R', 'B', 'N', 'P', '.']
        self.notation = ""
        tmp = len(self.pieces)
        self.pieces = {self.pieces[i]: i for i in range(len(self.pieces))}
        for keys in self.pieces:
            if keys == '.':
                self.pieces[keys] = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                self.pieces[keys] = (0, 0, 0) + (0, ) * self.pieces[keys] + (1, ) + (0, ) * (tmp - self.pieces[keys] - 1)
        try:
            with open("piece_value.json", "r") as readj:
                self.piece_value = json.load(readj)
        except FileNotFoundError:
            self.piece_value = {"Q": 9,"q": 9,"R": 5,"r": 5,"N": 3,"n": 3,"B": 3.2,"b": 3.2,"P": 1,"p": 1}

    def __str__(self) -> str:
        return f"FEN: {self.board.fen()}"

    def reset(self):
        self.board = chess.Board()
        self.n_games += 1
        self.notation = ""

    def set_position(self, newpos: str):
        self.board = chess.Board(newpos)

    def get_state(self):
        self.state = str(self.board)
        self.state = self.state.split("\n")
        for _ in range(8):
            self.state.extend(self.state.pop(0).split(" "))
        for _ in range(64):
            self.state.extend(self.pieces[self.state.pop(0)])
        return self.state

    def get_readable_state(self):
        self.state = str(self.board)
        return self.state

    def step(self, action):
        # action is -> [0, 0, 0, 1, 0, 0] where list is length of legal mvoes
        action = np.array(action, dtype='int16')
        action = np.argmax(action)
        reward = 0
        game_over = False
        move = list(self.board.legal_moves)[action]
        # ----------------------------------------------------------------
        # BEFORE MOVING PIECE
        # ----------------------------------------------------------------

        if self.board.is_capture(move):
            if self.board.is_en_passant(move):
                reward += 2
            else:
                reward += self.piece_value[str(self.board.piece_at(chess.parse_square(str(move)[2:4])))]

        if move.promotion is not None:
            reward += self.piece_value[str(move)[-1]]

        self.board.push(move)

        # ----------------------------------------------------------------
        # AFTER MOVE
        # ----------------------------------------------------------------

        if self.board.is_checkmate():
            reward += 240
            game_over = True
        elif (self.board.is_stalemate() or self.board.is_insufficient_material() or
              self.board.can_claim_fifty_moves() or self.board.can_claim_draw() or
              self.board.can_claim_threefold_repetition()):    # draws
            game_over = True

        self.notation += f"{action} "
        return reward, game_over


if __name__ == '__main__':
    print("Wrong File Idiot")