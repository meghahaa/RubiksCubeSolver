import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from rubiks_cube_env import RubiksCubeEnv, make_env  

# Constants
NUM_AGENTS = 6  # One for each face: R, L, F, B, U, D
MOVES_PER_AGENT = 2  # Each agent controls 2 moves (e.g., R, R')
TOTAL_MOVES = NUM_AGENTS * MOVES_PER_AGENT  # Total available moves across all agents
AGENT_NAMES = ['R', 'L', 'F', 'B', 'U', 'D']
AGENT_ACTIONS = {
    'R': [0, 1],     # R, R'
    'L': [2, 3],     # L, L'
    'F': [4, 5],     # F, F'
    'B': [6, 7],     # B, B'
    'U': [8, 9],     # U, U'
    'D': [10, 11],   # D, D'
}

# Hyperparameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
BATCH_SIZE = 64
UPDATE_TIMESTEPS = 2048
MAX_TIMESTEPS = 1000000
EVAL_INTERVAL = 10000
SOLVED_REWARD = 100  # Large positive reward when cube is solved
STEP_REWARD = -1     # Small negative reward for each step

# Neural Network Architecture for Actor and Critic
class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCritic, self).__init__()
        # Shared feature extractor
        self.feature_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.feature_network(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
        return action, action_probs.squeeze()[action].item(), value.item()
    
    def evaluate_actions(self, states, actions):
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, values.squeeze(-1), entropy

# Multi-Agent PPO Implementation
class MAPPO:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        
        # Get observation dimension from environment
        obs_dim = env.reset()[0].shape[0]
        
        # Initialize agents (each agent controls its own face moves)
        self.agents = {}
        for agent_name in AGENT_NAMES:
            self.agents[agent_name] = {
                'model': ActorCritic(obs_dim, MOVES_PER_AGENT).to(device),
                'optimizer': optim.Adam(
                    ActorCritic(obs_dim, MOVES_PER_AGENT).to(device).parameters(), 
                    lr=LEARNING_RATE
                )
            }
        
        # Setup TensorBoard logging
        self.writer = SummaryWriter(f'runs/MAPPO_Rubiks_Cube_{int(time.time())}')
        
        # Initialize metrics
        self.total_timesteps = 0
        self.episodes = 0
        self.best_reward = float('-inf')
    
    def select_action(self, state, deterministic=False):
        # Get action probabilities from each agent
        action_probs_dict = {}
        values_dict = {}
        
        for agent_name, agent_data in self.agents.items():
            model = agent_data['model']
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities and value from the agent's model
            action_probs, value = model(state_tensor)
            
            action_probs_dict[agent_name] = action_probs.squeeze().cpu().detach().numpy()
            values_dict[agent_name] = value.item()
        
        # Combine action probabilities from all agents
        combined_probs = np.zeros(TOTAL_MOVES)
        
        for agent_name, probs in action_probs_dict.items():
            for i, prob in enumerate(probs):
                global_action_idx = AGENT_ACTIONS[agent_name][i]
                combined_probs[global_action_idx] = prob
        
        # Normalize probabilities to sum to 1
        combined_probs = combined_probs / combined_probs.sum()
        
        # Sample action based on combined probabilities
        if deterministic:
            action = np.argmax(combined_probs)
        else:
            action = np.random.choice(np.arange(TOTAL_MOVES), p=combined_probs)
        
        # Determine which agent is responsible for this action
        responsible_agent = None
        local_action = None
        
        for agent_name, action_indices in AGENT_ACTIONS.items():
            if action in action_indices:
                responsible_agent = agent_name
                local_action = action_indices.index(action)
                break
        
        return action, responsible_agent, local_action, combined_probs[action], values_dict[responsible_agent]
    
    def collect_trajectories(self):
        # Storage for collected trajectories
        trajectories = {agent_name: {
            'states': [],
            'actions': [],
            'action_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        } for agent_name in AGENT_NAMES}
        
        # Reset environment
        state = self.env.reset()[0]
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Collect trajectories
        for t in range(UPDATE_TIMESTEPS):
            # Select action based on current state
            action, responsible_agent, local_action, action_prob, value = self.select_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Update episode metrics
            current_episode_reward += reward
            current_episode_length += 1
            
            # Store transition for the responsible agent
            trajectories[responsible_agent]['states'].append(state)
            trajectories[responsible_agent]['actions'].append(local_action)
            trajectories[responsible_agent]['action_probs'].append(action_prob)
            trajectories[responsible_agent]['rewards'].append(reward)
            trajectories[responsible_agent]['dones'].append(done)
            trajectories[responsible_agent]['values'].append(value)
            
            # Update state
            state = next_state
            
            # If episode is done, reset environment
            if done or truncated:
                self.episodes += 1
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                # Log episode metrics
                self.writer.add_scalar('metrics/episode_reward', current_episode_reward, self.episodes)
                self.writer.add_scalar('metrics/episode_length', current_episode_length, self.episodes)
                
                # Reset episode metrics
                current_episode_reward = 0
                current_episode_length = 0
                
                # Reset environment
                state = self.env.reset()[0]
        
        # Calculate advantages and returns for each agent
        for agent_name in AGENT_NAMES:
            # Compute GAE (Generalized Advantage Estimation)
            advantages = []
            returns = []
            
            # Get agent's trajectories
            agent_trajectories = trajectories[agent_name]
            
            if len(agent_trajectories['states']) == 0:
                # Skip if the agent didn't take any actions during this batch
                continue
            
            # Calculate last value for bootstrapping
            if agent_trajectories['dones'][-1]:
                last_value = 0
            else:
                last_state = agent_trajectories['states'][-1]
                with torch.no_grad():
                    _, last_value = self.agents[agent_name]['model'](
                        torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    )
                    last_value = last_value.item()
            
            # Initialize GAE
            gae = 0
            
            # Iterate in reverse order
            for i in reversed(range(len(agent_trajectories['rewards']))):
                # Calculate delta (TD error)
                if i == len(agent_trajectories['rewards']) - 1:
                    # For the last step, use bootstrapped value
                    next_value = last_value
                else:
                    next_value = agent_trajectories['values'][i + 1]
                
                delta = agent_trajectories['rewards'][i] + GAMMA * next_value * (1 - agent_trajectories['dones'][i]) - agent_trajectories['values'][i]
                
                # Update GAE
                gae = delta + GAMMA * GAE_LAMBDA * (1 - agent_trajectories['dones'][i]) * gae
                
                # Calculate return
                returns.insert(0, gae + agent_trajectories['values'][i])
                advantages.insert(0, gae)
            
            # Store advantages and returns
            agent_trajectories['advantages'] = advantages
            agent_trajectories['returns'] = returns
        
        return trajectories, np.mean(episode_rewards) if episode_rewards else 0, np.mean(episode_lengths) if episode_lengths else 0
    
    def update_policy(self, trajectories):
        # Update policy for each agent
        for agent_name in AGENT_NAMES:
            # Get agent's trajectories
            agent_trajectories = trajectories[agent_name]
            
            if len(agent_trajectories['states']) == 0:
                # Skip if the agent didn't take any actions during this batch
                continue
            
            # Convert trajectories to tensors
            states = torch.FloatTensor(agent_trajectories['states']).to(self.device)
            actions = torch.LongTensor(agent_trajectories['actions']).to(self.device)
            old_action_probs = torch.FloatTensor(agent_trajectories['action_probs']).to(self.device)
            returns = torch.FloatTensor(agent_trajectories['returns']).to(self.device)
            advantages = torch.FloatTensor(agent_trajectories['advantages']).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Create mini-batches for training
            dataset = torch.utils.data.TensorDataset(states, actions, old_action_probs, returns, advantages)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # PPO update
            for _ in range(NUM_EPOCHS):
                for batch_states, batch_actions, batch_old_probs, batch_returns, batch_advantages in dataloader:
                    # Get current action probabilities and values
                    action_log_probs, values, entropy = self.agents[agent_name]['model'].evaluate_actions(batch_states, batch_actions)
                    
                    # Calculate ratios
                    old_log_probs = torch.log(batch_old_probs + 1e-10)
                    ratios = torch.exp(action_log_probs - old_log_probs)
                    
                    # Calculate surrogate losses
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                    
                    # Calculate policy and value losses
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, batch_returns)
                    entropy_loss = -entropy.mean()
                    
                    # Calculate total loss
                    loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                    
                    # Update network parameters
                    self.agents[agent_name]['optimizer'].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agents[agent_name]['model'].parameters(), MAX_GRAD_NORM)
                    self.agents[agent_name]['optimizer'].step()
                    
                    # Log metrics
                    self.writer.add_scalar(f'losses/{agent_name}_policy_loss', policy_loss.item(), self.total_timesteps)
                    self.writer.add_scalar(f'losses/{agent_name}_value_loss', value_loss.item(), self.total_timesteps)
                    self.writer.add_scalar(f'losses/{agent_name}_entropy', entropy_loss.item(), self.total_timesteps)
    
    def evaluate(self, num_episodes=10):
        total_rewards = []
        total_lengths = []
        solved_count = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < 100:  # Limit evaluation episodes to 100 steps
                action, _, _, _, _ = self.select_action(state, deterministic=True)
                state, reward, done, _,_ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done and self.env.is_solved():
                    solved_count += 1
            
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        solve_rate = solved_count / num_episodes
        
        self.writer.add_scalar('eval/avg_reward', avg_reward, self.total_timesteps)
        self.writer.add_scalar('eval/avg_length', avg_length, self.total_timesteps)
        self.writer.add_scalar('eval/solve_rate', solve_rate, self.total_timesteps)
        
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.save_models('best_model')
        
        return avg_reward, avg_length, solve_rate
    
    def save_models(self, suffix=''):
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for agent_name in AGENT_NAMES:
            torch.save(
                self.agents[agent_name]['model'].state_dict(),
                f'models/agent_{agent_name}_model_{suffix}.pt'
            )
    
    def train(self):
        print("Starting MAPPO training...")
        
        while self.total_timesteps < MAX_TIMESTEPS:
            # Collect trajectories
            trajectories, avg_episode_reward, avg_episode_length = self.collect_trajectories()
            
            # Update policy
            self.update_policy(trajectories)
            
            # Update timesteps
            batch_timesteps = sum(len(trajectories[agent_name]['states']) for agent_name in AGENT_NAMES)
            self.total_timesteps += batch_timesteps
            
            # Log metrics
            self.writer.add_scalar('metrics/avg_episode_reward', avg_episode_reward, self.total_timesteps)
            self.writer.add_scalar('metrics/avg_episode_length', avg_episode_length, self.total_timesteps)
            
            # Print progress
            print(f"Timesteps: {self.total_timesteps}, Episodes: {self.episodes}, Avg Reward: {avg_episode_reward:.2f}, Avg Length: {avg_episode_length:.2f}")
            
            # Evaluate agent periodically
            if self.total_timesteps % EVAL_INTERVAL == 0:
                avg_reward, avg_length, solve_rate = self.evaluate()
                print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Solve Rate: {solve_rate:.2f}")
                
                # Save models
                self.save_models(f'timestep_{self.total_timesteps}')
        
        # Save final models
        self.save_models('final')
        self.writer.close()
        print("Training complete!")

# Main execution
if __name__ == "__main__":
    # Create Rubik's Cube environment
    env = make_env(max_steps_per_episode=100, scramble_moves=1)
    
    # Create MAPPO agent
    mappo = MAPPO(env)
    
    # Train agent
    mappo.train()