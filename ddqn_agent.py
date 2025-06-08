import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from neural_network import DuelingDQN, ReplayBuffer

class DDQNAgent:
    """
    DDQN智能體，使用雙重Q學習和經驗回放
    """
    def __init__(self, state_size, action_size, hyperparams):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 超參數
        self.lr = hyperparams.get('learning_rate', 0.001)
        self.gamma = hyperparams.get('gamma', 0.99)
        self.epsilon = hyperparams.get('epsilon', 1.0)
        self.epsilon_min = hyperparams.get('epsilon_min', 0.01)
        self.epsilon_decay = hyperparams.get('epsilon_decay', 0.995)
        self.batch_size = hyperparams.get('batch_size', 32)
        self.update_frequency = hyperparams.get('update_frequency', 1000)

        # 神經網絡
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # 經驗回放
        buffer_size = hyperparams.get('replay_buffer_size', 10000)
        self.memory = ReplayBuffer(buffer_size)

        # 計數器
        self.step_count = 0

        # 初始化目標網絡
        self.update_target_network()

    def update_target_network(self):
        """更新目標網絡"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.memory.push(state, action, reward, next_state, done)

    def select_action(self, state, training=True):
        """選擇動作（ε-貪婪策略）"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.cpu().data.numpy().argmax()

    def replay(self):
        """經驗回放訓練"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 採樣批次經驗
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones = batch

        # 轉換為張量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 當前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 雙重DQN：使用主網絡選擇動作，目標網絡評估Q值
        next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 計算損失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新ε值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期更新目標網絡
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.update_target_network()

        return loss.item()

    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load(self, filepath):
        """加載模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
