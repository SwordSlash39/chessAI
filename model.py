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
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout(p=0.1),
            nn.LeakyReLU()
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
            self.image_brain.append(nn.Dropout(p=0.1))
            self.image_brain.append(nn.LeakyReLU())
        self.process_brain = nn.Sequential(
            nn.Linear(16*8*8*16, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU()
        )
        for _ in range(2):
            self.process_brain.append(nn.Linear(4096, 4096))
            self.process_brain.append(nn.ReLU())
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

    def train_step(self, state, _, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        state = state.reshape((state.size(dim=0), 1, 8, 8, 16))
        next_state = next_state.reshape((next_state.size(dim=0), 1, 8, 8, 16))

        for i in range(state.size(dim=0)):
            pred = self.model(state[i])

            target = pred.clone()
            Q_new = reward[i]
            if not done[i]:
                unsq = next_state[i]
                Q_new = reward[i] + self.gamma * torch.max(self.model(unsq))
            target[0] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()
        if len(done) > 1:
            print(f"Trained off {len(done)} states!")

if __name__ == '__main__':
    print("Wrong File Idiot")