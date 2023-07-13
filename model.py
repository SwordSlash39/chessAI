import torch
import torch.nn as nn
import torch.optim as optim
import os

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # upstairs is testing
        self.image_brain = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16)
        )
        for _ in range(4):
            self.image_brain.append(
                nn.Conv3d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=3,
                    padding=1
                )
            )
            self.image_brain.append(nn.BatchNorm3d(16))
        self.process_brain = nn.Sequential(
            nn.Linear(16*8*8*16, 8192),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 4096),
            nn.BatchNorm1d(4096)
        )
        for _ in range(2):
            self.process_brain.append(nn.Linear(4096, 4096))
            self.process_brain.append(nn.BatchNorm1d(4096))
        self.process_brain.append(nn.Linear(4096, 1))
    def forward(self, x):
        x = self.image_brain(x)
        x = torch.reshape(x, (-1,))
        x = self.process_brain(x)
        return x

    # def save(self, file_name='model.pth'):
    #     model_folder_path = './model'
    #     if not os.path.exists(model_folder_path):
    #         os.makedirs(model_folder_path)

    #     file_name = os.path.join(model_folder_path, file_name)
    #     torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, rand, reward, next_state, done, color):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            rand = (rand, )
            color = (color, )
        state = state.reshape((state.size(dim=0), 1, 8, 8, 16))
        next_state = next_state.reshape((next_state.size(dim=0), 1, 8, 8, 16))
        priority = []

        for i in range(state.size(dim=0)):
            pred = self.model(state[i])
            target = pred.clone()
            Q_new = reward[i]
            if not done[i]:
                unsq = next_state[i]
                Q_new = reward[i] + self.gamma * self.model(unsq)[0]    # self.model returns 1 value so max or min dosent matter
            if not (rand and ((Q_new < pred[0] and color[i] == "white") or (rand and Q_new > pred[0] and color[i] == "black"))): # Do training as random exploration leaded to better results (if random and Q values less than prediction means random sucked ass)
                target[0] = Q_new
                self.optimizer.zero_grad()
                loss = self.criterion(target, pred)
                loss.backward()
                self.optimizer.step()
                if loss.item() > 4:
                    priority.append((torch.reshape(state[i], (-1,)).tolist(), rand[i], reward.tolist()[i], torch.reshape(next_state[i], (-1,)).tolist(), done[i], color[i]))
        if len(done) > 1:
            print(f"Trained off {len(done)} states!\n")
        return priority

if __name__ == '__main__':
    print("Wrong File Idiot")
