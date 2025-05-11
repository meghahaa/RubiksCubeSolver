
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.rubiks_cube_env import RubiksCubeEnv

def evaluate(model_name, model,scramble_moves, num_episodes):
    '''
    Evaluate the model on the Rubik's Cube environment. Print the success rate, average steps, and average rewards over the specified number of episodes.

    @param model_name: Name of the model (e.g., "DQN", "PPO").
    @param model: The trained model to evaluate.
    @param scramble_moves: Number of scrambles to perform on the cube during reset.
    @param num_episodes: Number of episodes to evaluate the model.
    '''

    print("+++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Evaluating {model_name} with Scramble {scramble_moves} averaging over {num_episodes} episodes")
    env = RubiksCubeEnv(scramble_moves=scramble_moves)

    episode_rewards = []
    episode_steps = []
    success_count = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        if done:
            success_count += 1

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    success_rate = success_count / num_episodes
    avg_steps = sum(episode_steps) / num_episodes
    avg_rewards = sum(episode_rewards) / num_episodes
    
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Rewards: {avg_rewards:.2f}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++")

    return success_rate, avg_steps, avg_rewards
    
