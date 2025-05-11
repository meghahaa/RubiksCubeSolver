import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import DQN
from envs.rubiks_cube_env import make_env
from evaluation.single_agent import evaluate

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

config = {
    "tensorboard_log": "./results/logs/tensorboard/",
    "monitor_log": "./results/logs/monitor/",
    "model_save_path": "./results/models/",
    "1_scramble_away_total_timesteps": 200000,
    "2_scramble_away_total_timesteps": 1500000,
    "3_scramble_away_total_timesteps": 5000000,
}

'''
Traning a DQN agent on the Rubik's Cube environment with 1 scramble away.
'''
print("=================================================")
print("1 move away from solved state")
print("=================================================")
env_1 = DummyVecEnv([lambda: make_env(scrambles_away=1, max_steps_per_episode=100)])
env_1 = VecMonitor(env_1, filename=os.path.join(config["monitor_log"], f"monitor_dqn_1"))

print("Creating DQN agent for 1 scramble away...")
model = DQN("MlpPolicy", env_1, verbose=1, tensorboard_log=config["tensorboard_log"])
print("Training DQN agent for 1 scramble away...")
with HiddenPrints():
    model.learn(total_timesteps=config["1_scramble_away_total_timesteps"])

print("Evaluating DQN agent for 1 scramble away...")
evaluate("DQN", model, 1, 100)

print("Saving DQN agent for 1 scramble away...")
model.save(os.path.join(config["model_save_path"], f"DQN_1.zip"))

env_1.close()

'''
Training a DQN agent on the Rubik's Cube environment with 2 scrambles away.
'''
print("=================================================")
print("2 moves away from solved state")
print("=================================================")
env_2 = DummyVecEnv([lambda: make_env(scrambles_away=2, max_steps_per_episode=100)])
env_2 = VecMonitor(env_2, filename=os.path.join(config["monitor_log"], f"monitor_dqn_2"))

print("Loading DQN agent for 1 scramble away for further learning on 2 scrambles away...")
model = DQN.load(os.path.join(config["model_save_path"], f"DQN_1.zip"), env=env_2)
print("Training DQN agent for 2 scrambles away...")
with HiddenPrints():
    model.learn(total_timesteps=config["2_scramble_away_total_timesteps"])

print("Evaluating DQN agent for 2 scrambles away...")
evaluate("DQN", model, 2, 100)

print("Saving DQN agent for 2 scrambles away...")
model.save(os.path.join(config["model_save_path"], f"DQN_2.zip"))
env_2.close()

'''
Training a DQN agent on the Rubik's Cube environment with 3 scrambles away.
'''
print("=================================================")
print("3 moves away from solved state")
print("=================================================")
env_3 = DummyVecEnv([lambda: make_env(scrambles_away=3, max_steps_per_episode=100)])
env_3 = VecMonitor(env_3, filename=os.path.join(config["monitor_log"], f"monitor_dqn_3"))

print("Loading DQN agent for 2 scrambles away for further learning on 3 scrambles away...")
model = DQN.load(os.path.join(config["model_save_path"], f"DQN_2.zip"), env=env_3)
print("Training DQN agent for 3 scrambles away...")
with HiddenPrints():
    model.learn(total_timesteps=config["3_scramble_away_total_timesteps"])

print("Evaluating DQN agent for 3 scrambles away...")
evaluate("DQN", model, 3, 100)

print("Saving DQN agent for 3 scrambles away...")
model.save(os.path.join(config["model_save_path"], f"DQN_3.zip"))
env_3.close()

