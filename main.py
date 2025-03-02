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
MAX_EPISODES = 1000      # Training episodes
GAMMA = 0.99                # Discount factor
LEARNING_RATE = 1e-3        # Learning rate for optimizer
BATCH_SIZE = 64             # Number of samples for training
MEMORY_SIZE = 50000        # Replay buffer size
EPSILON_START = 1.0         # Initial exploration probability
EPSILON_END = 0.01          # Minimum exploration probability
EPSILON_DECAY = 0.995       # Decay rate of epsilon
TARGET_UPDATE_FREQ = 100    # Target network update frequency
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

# ---- Q-Network for Pixel Input ----
class QNetworkPixel(nn.Module):
    """
    A simplified CNN similar to DeepMind's Atari DQN architecture.
    Expects input of shape (N, in_channels, 84, 84).
    """
    def __init__(self, in_channels, action_dim):
        super(QNetworkPixel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the output of the last conv layer:
        # For 84x84 input: 
        #   conv1 => (84-8)/4+1=20 -> (20-4)/2+1=9 -> (9-3)/1+1=7 => 64*7*7=3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # (N, 32, 20, 20)
        x = torch.relu(self.conv2(x))   # (N, 64, 9, 9)
        x = torch.relu(self.conv3(x))   # (N, 64, 7, 7)
        x = x.view(x.size(0), -1)       # flatten to (N, 3136)
        x = torch.relu(self.fc1(x))     # (N, 512)
        return self.fc2(x)              # (N, action_dim)

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
def train_ddqn(env_name="CartPole-v1", episodes=MAX_EPISODES):
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

        logger.info(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")
    plt.plot(total_rewards)
    plt.savefig(f"output_{env_name}.png")

    with open(f"rewards_{env_name}.json", "w") as f:
        f.write(json.dumps(total_rewards))

    agent.memory.done()
    env.close()



class DDQNAgentPixel:
    def __init__(self, in_channels, action_dim):
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = LocalReplayBuffer(MEMORY_SIZE)

        self.q_network = QNetworkPixel(in_channels, action_dim).to(device)
        self.target_network = QNetworkPixel(in_channels, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        self.update_target_network()
        self.total_steps = 0  # count environment steps

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        """
        state: Numpy array of shape (1, 84, 84).
        We'll unsqueeze(0) to get (batch=1, channels=1, H=84, W=84).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device) 
            # shape => (1, 1, 84, 84)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            return q_values.argmax().item()

    def train_step(self):
        # Only train if we have enough samples
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        # Convert to PyTorch tensors
        states_t      = torch.FloatTensor(states).to(device)       # (B, 1, 84, 84)
        actions_t = torch.LongTensor(actions).view(-1, 1).to(device)     # (B, 1)
        rewards_t     = torch.FloatTensor(rewards).to(device)      # (B, 1)
        next_states_t = torch.FloatTensor(next_states).to(device)  # (B, 1, 84, 84)
        dones_t       = torch.FloatTensor(dones).to(device)        # (B, 1)

        # Current Q-values
        q_values = self.q_network(states_t).gather(1, actions_t)

        # DDQN target
        with torch.no_grad():
            # next action from current Q-net
            next_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            # evaluate using target network
            next_q_values = self.target_network(next_states_t).gather(1, next_actions)
            target = rewards_t + GAMMA * (1 - dones_t) * next_q_values

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # Exponential decay or any other schedule
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def preprocess(obs):
    """
    Convert (210, 160, 3) to (1, 84, 84) grayscale and scale to [0,1].
    """
    # obs is typically uint8 [0..255]
    # 1) convert to grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

    # 2) resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    # 3) normalize to [0,1]
    normalized = resized.astype(np.float32) / 255.0

    # 4) add channel dimension => (1, 84, 84)
    return np.expand_dims(normalized, axis=0)

def train_ddqn_pixel(env_name = "Pong-v0", episodes=MAX_EPISODES):
    env = gym.make("Pong-v0")
    action_dim = env.action_space.n  # Typically 6 for Pong
    in_channels = 1                  # We'll do grayscale input
    agent = DDQNAgentPixel(in_channels, action_dim)

    rewards_history = []
    best_mean_reward = float('-inf')

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess(obs)      # shape (1, 84, 84)
        total_reward = 0

        done = False
        t = 0
        while not done:
            action = agent.select_action(state)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = preprocess(obs_next)
            agent.memory.push(state, action, reward, next_state, done)
            total_reward += reward

            # Train step
            agent.train_step()

            # Update state
            state = next_state
            t += 1
            agent.total_steps += 1

            # Update target network periodically
            if agent.total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # Epsilon decay each step
            agent.update_epsilon()

        rewards_history.append(total_reward)

        # Logging
        mean_reward_100 = np.mean(rewards_history[-100:])
        print(f"Episode {ep+1} | Ep.Reward: {total_reward:.2f} | "
              f"Mean(100): {mean_reward_100:.2f} | Epsilon: {agent.epsilon:.3f}")


    env.close()
    plt.plot(rewards_history)
    plt.savefig(f"output_{env_name}.png")

    with open(f"rewards_{env_name}.json", "w") as f:
        f.write(json.dumps(total_rewards))

    agent.memory.done()
    env.close()




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        # Define layers
        self.fc1 = nn.Linear(state_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, action_dim)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize the output layer weights to small values
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        # Forward pass with LeakyReLU activation
        x = torch.nn.functional.leaky_relu(self.ln1(self.fc1(state)), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        return torch.tanh(self.fc3(x)) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # State pathway
        self.fc1_state = nn.Linear(state_dim, 400)
        self.ln1_state = nn.LayerNorm(400)
        # Action pathway
        self.fc1_action = nn.Linear(action_dim, 400)
        # Combined pathway
        self.fc2 = nn.Linear(800, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize the output layer weights to small values
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state, action):
        # Forward pass for state and action
        s_out = torch.nn.functional.leaky_relu(self.ln1_state(self.fc1_state(state)), negative_slope=0.01)
        a_out = torch.nn.functional.leaky_relu(self.fc1_action(action), negative_slope=0.01)
        # Combine state and action features
        x = torch.cat([s_out, a_out], dim=1)
        x = torch.nn.functional.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        return self.fc3(x)

class MultiAgentDDPG:
    def __init__(self, state_dim, action_dim, env):
        max_action = env.action_space.high[0]  # Get max action value
        self.state_dim = state_dim
        self.action_dim = action_dim
        if REDIS_HOST is None:
            self.memory = LocalReplayBuffer(MEMORY_SIZE)
        else:
            self.memory = SharedReplayBuffer(MEMORY_SIZE)
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, max_action).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        # Noise for exploration
        self.noise = OrnsteinUhlenbeckNoise(action_dim)

        # Initialize target networks
        self.update_target_networks()

    def update_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action + self.noise()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Update Critic
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, next_actions)
        target_q_values = rewards.unsqueeze(1) + GAMMA * target_q_values * (1 - dones.unsqueeze(1))
        critic_loss = nn.MSELoss()(self.critic(states, actions), target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data = target_param.data * 0.995 + param.data * 0.005
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data = target_param.data * 0.995 + param.data * 0.005


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.mu))
        self.state = self.state + dx
        return self.state

import json


def train_single_agent_ddpg(env_name="Pendulum-v1", episodes=MAX_EPISODES):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MultiAgentDDPG(state_dim, action_dim, env)
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)

            agent.train()
            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)

        # Log reward
        logger.info(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")

        with open(f"rewards_{env_name}.json", "w") as f:
            json.dump(total_rewards, f)

    env.close()
# Run the training
if __name__ == "__main__":
    train_single_agent_ddpg()
