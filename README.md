# RubiksCubeSolver

This repository contains the code for training and evaluating reinforcement learning agents to solve a Rubik's Cube. We have implemented and compared three different approaches: single agent learning, multi-agent learning, and sequential agent learning.

## Implementation Details

Our implementation is structured into the following key components:

-   **envs**: This folder contains the code for our custom Rubik's Cube environments.
    -   `rubiks_cube_env.py`: This file defines the base `RubiksCubeEnv` environment. It provides the fundamental Rubik's Cube simulation and is used as the foundation for training single agents.
    -   `f2l_agent_env.py` and `cross_agent_env.py`: These files define wrapper environments built on top of `RubiksCubeEnv`. They are specifically designed to facilitate the sequential learning process for the F2L and Cross agents, respectively.

-   **agents**: This folder houses the training scripts for all our reinforcement learning agents. We utilize curriculum learning to train these agents.
    -   `dqn_agent.py`: Contains the training code for a single-agent using the Deep Q-Network (DQN) algorithm.
    -   `ppo_agent.py`: Contains the training code for a single-agent using the PPO algorithm.
    -   `mappo_agent.py`: Contains the training code for the Multi-Agent PPO (MAPPO) approach.
    -   `cross_ppo_agent.py`: Contains the training code for the Cross agent using the Proximal Policy Optimization (PPO) algorithm.
    -   `f2l_ppo_agent.py`: Contains the training code for the First Two Layers (F2L) agent using the PPO algorithm.
    
-   **evaluation**: This folder contains the evaluation script used to assess the performance of the trained agents.
    -   `single_agent.py`: This script includes the `evaluate` function, which takes a trained agent and evaluates its performance over a specified number of episodes. The evaluation metrics include success rate, average steps taken to solve, and average rewards received.

-   **results**: This folder stores the outcomes of the training process. It has two subfolders:
    -   **logs/tensorboard**: This directory contains the TensorBoard log files generated during the training of each model. These files can be used to visualize and track various training metrics, such as loss, reward, and episode length, in TensorBoard. The logs are organized into subdirectories for each agent and training run (e.g., `cross/ppo1/PPO_0`).
    -   **models**: This directory stores the saved model checkpoints of the trained agents. These models can be loaded for further training or for evaluation. The models are typically saved as `.zip` files, with subdirectories organizing models by agent type (e.g., `cross`, `f2l`).
-   `example_nb.ipynb`: This Jupyter Notebook provides an example of how to render the custom `RubiksCubeEnv`. It demonstrates loading a trained PPO agent and visualizing its attempt to solve a Rubik's Cube scrambled 3 moves away.

## Usage

**Note**: You need to run all scripts from the root directory of the repository, which is `RubiksCubeSolver`.

**Install all python dependencies:**

```bash
pip install -r requirements.txt
```

**To train the agents:**

Run the corresponding training script from the `agents` directory.

```bash
python3 agents/{desired agent file}
```

**Example:** To train the PPO single agent:

```bash
python3 agents/ppo_agent.py
```
**Note**: We have already included the trained models and their corresponding logs in this repository. If you wish to directly evaluate the pre-trained models or visualize their training progress, you can skip the training commands.

**To visualize training results:**

Use TensorBoard to visualize the training metrics.

```bash
tensorboard --logdir=results/logs/tensorboard/
```

Then, open your web browser and navigate to the address provided by TensorBoard (usually `http://localhost:6006/`).

**example\_nb.ipynb:**

This Jupyter Notebook demonstrates how to load a trained PPO agent and visualize its solving process on a Rubik's Cube scrambled 3 moves away. You can open and run this notebook using Jupyter:

```bash
jupyter notebook example_nb.ipynb
```
