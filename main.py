import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt
np.bool8 = np.bool_
np.float_ = np.float64

# Hyperparameters
MAX_EPISODES = 2000
GAMMA = 0.99
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 500


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, int(done)))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            np.stack(states),
            np.stack(actions),
            np.stack(rewards),
            np.stack(next_states),
            np.stack(dones),
        )

    def __len__(self):
        return len(self.buffer)

# DDQN Agent
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            target_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + GAMMA * target_q_values * (1 - dones.unsqueeze(1))

        # Loss and optimization
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, step):
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY)

# Main Training Loop
def train_ddqn(env_name="Acrobot-v1", episodes=MAX_EPISODES):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    total_steps = 0
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train()
            agent.update_epsilon(total_steps)

            if done:
                break

            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
        total_rewards.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
    plt.plot(total_rewards)
    plt.show()

    env.close()

# Run the training
if __name__ == "__main__":
    train_ddqn()
