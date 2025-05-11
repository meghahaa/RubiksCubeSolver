import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
from envs.cross_agent_env import make_env
from evaluation.single_agent import evaluate

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

config = {
    "tensorboard_log": "./results/logs/tensorboard/cross/",
    "monitor_log": "./results/logs/monitor/cross/",
    "model_save_path": "./results/models/cross/",
    "initial_total_timesteps": 200000,
}

def train_cross_ppo_agent(max_scramble_moves):
    """
    Train a PPO agent on the CrossAgent environment with a specified number of scrambles away from the solved state.
    Starting from 1 scramble away and increasing the number of scrambles after each training phase.
    """
    for scramble_moves in range(3, max_scramble_moves + 1):
        print("=================================================")
        print(f"{scramble_moves} move(s) away from solved state")
        print("=================================================")

        tb_log_name = f"ppo{scramble_moves}"

        # Create the environment
        env = DummyVecEnv([lambda: make_env(scrambles_away=scramble_moves, max_steps_per_episode=100)])
        env = VecMonitor(env, filename=os.path.join(config["monitor_log"], f"monitor_cross_ppo_{scramble_moves}"))

        if scramble_moves == 1:
            print(f"Creating PPO agent for {scramble_moves} scramble(s) away...")
            model = PPO("MlpPolicy", env, verbose=1, 
                        tensorboard_log=os.path.join(config["tensorboard_log"], tb_log_name),
                        device='cpu')
        else:
            print(f"Loading PPO agent for {scramble_moves-1} scramble away for further learning on {scramble_moves} scrambles away...")
            model = PPO.load(
                os.path.join(config["model_save_path"], f"cross_PPO_{scramble_moves-1}.zip"),
                env=env,
                device='cpu',
                custom_objects={
                    "tensorboard_log": os.path.join(config["tensorboard_log"], tb_log_name)
                }
            )
        max_timesteps = config["initial_total_timesteps"] if scramble_moves == 1 else config["initial_total_timesteps"] * (scramble_moves ** 2)

        print(f"Training PPO agent for {scramble_moves} scramble(s) away...")
        while True:
            try:
                with HiddenPrints():
                    model.learn(total_timesteps=max_timesteps, reset_num_timesteps=False)

                print(f"Evaluating PPO agent for {scramble_moves} scramble(s) away...")
                sucess,avg_steps,_= evaluate("PPO", model, scramble_moves, 100)

                if sucess > 0.85:
                    break
            except KeyboardInterrupt:
                print("Training interrupted. Exiting...")
                break

        print(f"Saving PPO agent for {scramble_moves} scramble(s) away...")
        model.save(os.path.join(config["model_save_path"], f"cross_PPO_{scramble_moves}.zip"))

        env.close()

if __name__ == "__main__":
    max_scramble_moves = 3  # Set the maximum number of scrambles away
    print("TRAINING CROSS PPO AGENT WITH SCRAMBLE MOVES: ",max_scramble_moves)
    train_cross_ppo_agent(max_scramble_moves)

