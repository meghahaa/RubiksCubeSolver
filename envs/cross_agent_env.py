from envs.rubiks_cube_env import RubiksCubeEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

class CrossAgentEnv(RubiksCubeEnv):
    """
    A subclass of RubiksCubeEnv that focuses on solving the bottom cross of a Rubik's Cube.
    """
    def __init__(self, scramble_moves):
        super().__init__(scramble_moves)
        self.edge_pairs = [
            (46, 25, 22),  # Front
            (50, 34, 31),  # Right
            (52, 43, 40),  # Back
            (48, 16, 13)   # Left
        ]
        print("CrossAgentEnv initialized with scramble_moves:", scramble_moves)


    def is_solved(self):
        """
        Checks if the bottom cross is solved:
        - All 4 yellow cross edge stickers are correct on the bottom face.
        - Their adjacent stickers on the side faces align with the center of that face.
        """
        bottom_center = self.state[49]  # Yellow center

        for bottom_idx, side_idx, center_idx in self.edge_pairs:
            if self.state[bottom_idx] != bottom_center:
                return False
            if self.state[side_idx] != self.state[center_idx]:
                return False

        return True
    
    def calculate_similarity_reward(self, state):
        """
        Compute the similarity reward based on the current state.
        The reward is calculated based on number of bottom cross pieces correctly placed.
        The more pieces that match, the higher the reward.
        """

        bottom_center = self.state[49]  # Yellow center
        correct_pieces = 0
        for bottom_idx, side_idx, center_idx in self.edge_pairs:
            if self.state[bottom_idx] == bottom_center:
                correct_pieces += 1
            if self.state[side_idx] == self.state[center_idx]:
                correct_pieces += 1
        return correct_pieces / 8


    def calculate_reward(self, prev_state, current_state):
        """
        Calculate the reward based on the change in the number of correctly placed bottom cross pieces.
        The reward is shaped to encourage the agent to make progress towards solving the bottom cross.
        """
        prev_reward = self.calculate_similarity_reward(prev_state)
        current_reward = self.calculate_similarity_reward(current_state)
        reward = current_reward - prev_reward
        reward=reward*10 - 0.1
        if current_reward == 1:
            reward = 100
        return reward

    
def make_env(scrambles_away,max_steps_per_episode=100):
    '''
    Create and return CrossAgentEnv with specified parameters.

    @param scrambles_away: Number of scrambles to perform on the cube during reset.
    @param max_steps_per_episode: Maximum number of steps per episode.
    '''
    env = CrossAgentEnv(scramble_moves=scrambles_away)

    # Set the maximum number of steps per episode
    env = TimeLimit(env, max_episode_steps=max_steps_per_episode)
    # Monitor the environment to log episode stats
    env = Monitor(env)
    return env

# env=CrossAgentEnv(scramble_moves=2)
# print(env.is_solved())
# env.render()
