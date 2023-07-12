import torch, chess, os, random
from collections import deque
from model import Linear_QNet, QTrainer, DEVICE
from gameChess import chess_game

MAX_MEMORY = 80_000
BATCH_SIZE = 2_500
LR = 0.001

class Agent:
    def __init__(self, testing=False):
        self.testing = testing
        self.epsilon = 0
        self.gamma = 0.9
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet().to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        try:
            path = "model/model.pth"
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model"])
            if not testing: # if testing its a waste of resources to load optimizer (no training done)
                self.n_games = checkpoint["games"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Model at {path} loaded")
        except RuntimeError:
            pass
        except FileNotFoundError:
            pass
    
    def saveModel(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save({
            "games": self.n_games,
            'model': self.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict()
            }, file_name)
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train_long_memory(self, rand):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, rand)

    def train_short_memory(self, state, action, reward, next_state, done, rand):
        self.trainer.train_step(state, action, reward, next_state, done, rand)

    def get_action(self, board: chess.Board, color):
        self.epsilon = (80 * (0.99 ** self.n_games) + 1) if not self.testing else -1
        legalMoves = list(board.legal_moves)
        tmp = chess_game()
        output = [0] * len(legalMoves)
        
        if len(legalMoves) == 1:    # optimize for forced moves
            return [1]
        
        if random.random() * 200 < self.epsilon:
            output[random.randint(0, len(legalMoves) - 1)] = 1
            return output, True

        pos_eval = []
        for i in range(len(legalMoves)):
            board.push(legalMoves[i])
            tensorState = torch.tensor(tmp.get_state(), dtype=torch.float, device=DEVICE)
            tensorState = tensorState.reshape((8, 8, 16))
            tensorState = tensorState.unsqueeze(0)
            with torch.no_grad():
                pos_eval.append(self.model(tensorState)[0])
            # remove last temporary move
            board.pop()
        move = torch.tensor(pos_eval, dtype=torch.float16, device=DEVICE)
        if color == "white":
            output[move.argmax().item()] = 1
            return output, False
        output[move.argmin().item()] = 1
        return output, False


if __name__ == '__main__':
    print("Wrong File Idiot")
