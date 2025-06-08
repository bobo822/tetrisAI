import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DuelingDQN(nn.Module):
    """
    雙重Q網絡架構，將狀態值和優勢值分開估計
    實現論文: Dueling Network Architectures for Deep Reinforcement Learning
    """
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # 特徵提取層
        self.feature_layer = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 價值流
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 優勢流
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        # 分別計算價值和優勢
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q值 = 價值 + (優勢 - 平均優勢)
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return qvals

class ReplayBuffer:
    """
    經驗回放緩衝區，用於存儲和採樣經驗
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)
