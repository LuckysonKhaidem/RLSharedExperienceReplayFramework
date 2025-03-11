# import gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import json
# np.bool8 = bool

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size=128):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.actor = nn.Linear(hidden_size, action_dim)
#         self.critic = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         policy_logits = self.actor(x)
#         value = self.critic(x)
#         return policy_logits, value

# def compute_returns(rewards, gamma):
#     """
#     Computes the discounted returns for each timestep in an episode.
#     """
#     returns = []
#     G = 0
#     for r in reversed(rewards):
#         G = r + gamma * G
#         returns.insert(0, G)
#     return returns

# def train_actor_critic(env_name="CartPole-v1", 
#                        hidden_size=128, 
#                        lr=1e-3, 
#                        gamma=0.99, 
#                        max_episodes=1000):
    
#     # Initialize environment, network, and optimizer
#     env = gym.make(env_name)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
    
#     model = ActorCritic(state_dim, action_dim, hidden_size)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     total_rewards = []
#     for episode in range(max_episodes):
#         state, _ = env.reset()
#         done = False
        
#         states = []
#         actions = []
#         rewards = []
#         log_probs = []
#         values = []

#         # --- Collect one episode ---
#         while not done:
#             state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
#             logits, value = model(state_t)

#             # Get action probabilities
#             probs = F.softmax(logits, dim=-1)
#             # Sample an action
#             action_dist = torch.distributions.Categorical(probs)
#             action = action_dist.sample()

#             # Log probability of that action (for the policy gradient)
#             log_prob = action_dist.log_prob(action)
#             next_state, reward, done, _, _ = env.step(action.item())

#             # Store transition
#             states.append(state)
#             actions.append(action.item())
#             rewards.append(reward)
#             log_probs.append(log_prob)
#             values.append(value.squeeze(0))

#             state = next_state

#         # --- Compute returns ---
#         returns = compute_returns(rewards, gamma)
#         returns = torch.FloatTensor(returns)

#         # --- Calculate losses ---
#         policy_loss = []
#         value_loss = []
#         for log_prob, R, value_est in zip(log_probs, returns, values):
#             advantage = R - value_est
#             policy_loss.append(-log_prob * advantage)
#             value_loss.append(F.mse_loss(value_est, R))

#         # Sum up the losses
#         loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

#         # --- Gradient descent step ---
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # --- Logging ---
#         total_reward = sum(rewards)
#         total_rewards.append(total_reward)
#         print(f"Episode {episode}, total reward: {total_reward:.2f}")

#     env.close()
#     with open(f"rewards_a2c_{env_name}_local.json", "w") as f:
#         f.write(json.dumps(total_rewards))

# if __name__ == "__main__":
#     train_actor_critic()



import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json

# For newer NumPy versions that have deprecated np.bool, you did:
np.bool8 = bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head: outputs mean for each action dimension
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        # We keep a log-std parameter for each action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head: outputs a single state-value
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        
        # Actor: mean and (log-)standard deviation
        mean = self.actor_mean(x)
        std = self.log_std.exp().expand_as(mean)
        
        # Critic: state value
        value = self.critic(x)
        
        return mean, std, value

def compute_returns(rewards, gamma):
    """
    Computes the discounted returns for each timestep in an episode.
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def train_actor_critic(env_name="Pendulum-v1", 
                       hidden_size=128, 
                       lr=1e-3, 
                       gamma=0.99, 
                       max_episodes=200):
    
    # ---------------------------
    # 1) Make Pendulum-v1 env
    # ---------------------------
    env = gym.make(env_name)
    
    # ---------------------------
    # 2) Continuous action space
    #    shape[0] instead of .n
    # ---------------------------
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = ActorCritic(state_dim, action_dim, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_rewards = []
    
    for episode in range(max_episodes):
        # Gymnasium resets often return (obs, info)
        # So we do: 
        state, _ = env.reset()
        done = False
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        total = 0
        # --- Collect one episode ---
        for t in range(1000):
            # Torchify
            state_t = torch.FloatTensor(state).view(1, -1).to(device)
            
            # Forward pass
            mean, std, value = model(state_t)

            # ---------------------------
            # 3) Continuous action distribution
            # ---------------------------
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()  # sample from Normal
            
            # We need to sum log_prob across action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Clamp or scale action to [-2, 2] for Pendulum
            action_clamped = torch.clamp(action, -2.0, 2.0)
            
            # Convert to numpy
            action_np = action_clamped.cpu().detach().numpy()[0]
            
            # Step in environment
            next_state, reward, done, _, _ = env.step([action_np])

            # Store transition
            states.append(state)
            actions.append(action_np)
            rewards.append(reward.item())
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            
            state = next_state

        # --- Compute returns ---
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)

        # --- Calculate losses ---
        policy_loss = []
        value_loss = []
        for log_prob, R, value_est in zip(log_probs, returns, values):
            advantage = R - value_est
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.mse_loss(value_est, R))

        # Sum up the losses
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        # --- Gradient descent step ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Logging ---
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        print(f"Episode {episode}, total reward: {total_reward}")
    
    env.close()
    
    with open(f"rewards_a2c_{env_name}_local.json", "w") as f:
        f.write(json.dumps(total_rewards))

if __name__ == "__main__":
    train_actor_critic()

