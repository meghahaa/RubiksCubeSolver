import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.cross_agent_env import CrossAgentEnv

from envs.cross_agent_env import CrossAgentEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


class F2LAgentEnv(CrossAgentEnv):
    '''
    F2LAgentEnv is a subclass of CrossAgentEnv that focuses on solving the first two layers (F2L) of the Rubik's Cube.
    '''
    def __init__(self, scramble_moves):

        self.model = PPO.load(os.path.join('results/models', f"cross_PPO_3.zip"),device='cpu')

        self.f2l_check = {
            'bottom': [45,47,51,53],
            'front':  [21,24,23,26],
            'left':   [12,15,14,17],
            'right':  [30,33,32,35],
            'back':   [39,42,41,44]
        }

        self.centers = {
            'bottom': 49,
            'front':  22,
            'left':   13,
            'right':  31,
            'back':   40
        }
        self.edge_pairs = [
            (46, 25, 22),  # Front
            (50, 34, 31),  # Right
            (52, 43, 40),  # Back
            (48, 16, 13)   # Left
        ]

        super().__init__(scramble_moves)
        print('F2LAgentEnv initialized with scramble_moves:', scramble_moves)

    def is_solved(self):
        '''
        Checks if the first two layers (F2L) are solved:
        - All 4 F2L corner pieces are correctly placed.
        - The edge pieces adjacent to the F2L corners are aligned with the centers of their respective faces.
        '''
        # First check cross is done and aligned
        if not super().is_solved():
            return False

        # Check if F2L corners are correctly placed
        for face, indices in self.f2l_check.items():
            face_center = self.state[self.centers[face]]
            for idx in indices:
                if self.state[idx] != face_center:
                    return False
        return True
    
    def calculate_similarity_reward(self, state):
        """
        Compute the similarity reward based on the current state.
        The reward is calculated based on the number of correctly placed F2L corner pieces and aligned cross edges.
        The more pieces that match, the higher the reward.
        """
        reward = 0
        # Check if F2L corners are correctly placed
        for face, indices in self.f2l_check.items():
            face_center = self.state[self.centers[face]]
            for idx in indices:
                if self.state[idx] == face_center:
                    reward += 1 
        reward= reward / 20  # Normalize to 0-1   
        return reward
    
    def reset(self, *, seed=None, options=None):
        '''
        Reset the environment to a scrambled state but with cross already solved.
        '''
        obs = super().reset()[0]
        self.state = obs
        steps = 0
        done = False
        while not done and steps < 100:
            action, _ = self.model.predict(obs)
            obs, reward, done, truncated, info = self.step(action)
            steps += 1
        obs = self.get_observation()
        return obs, info
        
                        
def make_env(scrambles_away,max_steps_per_episode=100):
    '''
    Create and return CrossAgentEnv with specified parameters.

    @param scrambles_away: Number of scrambles to perform on the cube during reset.
    @param max_steps_per_episode: Maximum number of steps per episode.
    '''
    env = F2LAgentEnv(scramble_moves=scrambles_away)

    # Set the maximum number of steps per episode
    env = TimeLimit(env, max_episode_steps=max_steps_per_episode)
    # Monitor the environment to log episode stats
    env = Monitor(env)
    return env

# env=F2LAgentEnv(scramble_moves=2)
# print(env.is_solved())
# env.render()