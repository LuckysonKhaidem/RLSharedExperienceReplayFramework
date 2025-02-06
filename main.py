import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import deque
import simplejson as json
import redis
import sys
import os
import json

np.bool8 = np.bool_
np.float_ = np.float64

# Hyperparameters
MAX_EPISODES = 1000        # Number of training episodes
GAMMA = 0.99               # Discount factor (good for long-term reward optimization)
LEARNING_RATE = 1e-3       # Learning rate (reasonable for Q-learning/DQN)
BATCH_SIZE = 64            # Lowered batch size (128 might be too high for simple environments)
MEMORY_SIZE = 50000        # Reduced replay buffer size (400k is overkill for Taxi-v3)
EPSILON_START = 1.0        # Initial exploration probability
EPSILON_END = 0.05         # Minimum exploration probability (0.01 may be too low)
EPSILON_DECAY = 0.997      # Adjusted decay rate for smoother transition
TARGET_UPDATE_FREQ = 50    # More frequent target network updates for stability
REDIS_HOST = sys.argv[1] if len(sys.argv) > 1 else None  # Redis setup

COUNTER_KEY = "counter"

import logging

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler("run.log")  # Write logs to a file
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()  # Write logs to stdout
console_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add formatter to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)



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

class QNetworkPixel(nn.Module):
    def __init__(self, in_channels, action_dim):
        """
        Args:
            in_channels (int): Number of channels in the input image (e.g., 1 for grayscale or 3 for RGB).
            action_dim (int): Number of discrete actions in the environment.
        """
        super(QNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # After convolutional layers, we flatten before the linear layers
        # The input to the first linear layer depends on the image size.
        # For a typical 84x84 input, the last conv layer produces a 64 x 7 x 7 output => 64*7*7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size(0), -1)   # (batch_size, 64*7*7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Replay Buffer
class SharedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.r = redis.Redis(host=REDIS_HOST, port = 6379)
        self.r.incr(COUNTER_KEY)
        c = int(self.r.get(COUNTER_KEY))
        if (c == 1):
            self.r.delete("shared")

    def push(self, state, action, reward, next_state, done):
        data = { "state" : state, "action": action, "reward": reward, "next_state": next_state, "done" : int(done)}
        data = json.dumps(data, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj)
        self.r.rpush("shared", data)
        if (self.r.llen("shared") >= self.capacity):
            self.r.ltrim("shared", 1, -1)


    def sample(self, batch_size):
        size = self.r.llen("shared")
        if size == 0:
            return []
        random_indices = random.sample(range(size), min(batch_size, size))
        samples = []
        for index in random_indices:
            try:
                samples.append(json.loads(self.r.lindex("shared", index)))
            except:
                continue
        states = [sample["state"] for sample in samples]
        actions = [sample["action"] for sample in samples]
        rewards = [sample["reward"] for sample in samples]
        next_states = [sample["next_state"] for sample in samples]
        dones = [sample["done"] for sample in samples]
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)

    def done(self):
        self.r.decr(COUNTER_KEY)
        self.r.close()

    def __len__(self):
        return self.r.llen("shared")
    
# Replay Buffer
class LocalReplayBuffer:
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
    
    def done(self):
        pass

# DDQN Agent
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        if REDIS_HOST is None:
            self.memory = LocalReplayBuffer(MEMORY_SIZE)
        else:
            self.memory = SharedReplayBuffer(MEMORY_SIZE)

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
def train_ddqn(env_name="Taxi-v3", episodes=MAX_EPISODES):
    env = gym.make(env_name)
    if len(env.observation_space.shape) > 0:
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = 1
    action_dim = env.action_space.n
    agent = DDQNAgent(state_dim, action_dim)
    total_steps = 0
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            action = agent.select_action([state])
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push([state], action, reward, [next_state], done)
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

        logger.info(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
    plt.plot(total_rewards)
    plt.savefig(f"output_{env_name}.png")

    with open(f"rewards_{env_name}.json", "w") as f:
        f.write(json.dumps(total_rewards))

    agent.memory.done()
    env.close()

# Run the training
if __name__ == "__main__":
    train_ddqn()
