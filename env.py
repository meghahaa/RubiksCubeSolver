import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class RubiksCubeEnv(gym.Env):
    '''
    Custom environment for a Rubik's Cube using OpenAI Gym.
    '''

    def __init__(self):
        super(RubiksCubeEnv, self).__init__()
        
        '''
        Action space with 12 discrete actions: 
        - 6 faces (U, L, F, R, B, D)
        - Each face can be rotated clockwise or counter-clockwise
        - 6 * 2 = 12 actions
        - 0-5 for clockwise, 6-11 for counter-clockwise
        - 0: U, 1: L, 2: F, 3: R, 4: B, 5: D
        - 6: U', 7: L', 8: F', 9: R', 10: B', 11: D'
        '''
        self.action_space = spaces.Discrete(12)
        
        '''
        Observation space representing the state of the Rubik's Cube:
        - 1D flattened array of size 54 (6 faces * 9 cells per face)
        - 6 faces, each face is a 3x3 grid
        - Each cell can have values from 0 to 5 (representing colors)
        - 0: White, 1: Yellow, 2: Red, 3: Orange, 4: Green, 5: Blue
        - Shape: (54,)
        - dtype: int32
        '''
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.int32)
        
        '''
        Initial state of the Rubik's Cube
        '''
        self.state = self.get_solved_cube()

    def get_solved_cube(self):
        '''
        Returns a solved state of the Rubik's Cube.
        '''
        return np.array([
            0,0,0,0,0,0,0,0,0,  # Top (White)
            3,3,3,3,3,3,3,3,3,  # Left (Orange)
            4,4,4,4,4,4,4,4,4,  # Front (Green)
            2,2,2,2,2,2,2,2,2,  # Right (Red)
            5,5,5,5,5,5,5,5,5,  # Back (Blue)
            1,1,1,1,1,1,1,1,1   # Bottom (Yellow)
        ])

    def step(self, action):
        '''
        Perform an action on the Rubik's Cube.
        Action is an integer representing a move (e.g., 0-11 for 6 faces * 2 directions).
        '''

        prev_state = self.state.copy()
        self.rotate(action)
        reward = self.calculate_reward(prev_state, self.state)
        done = self.is_solved()
        
        return self.state, reward, done, {}

    def reset(self):
        '''
        Reset the environment to the initial state.'
        '''

        self.state = self.get_solved_cube()
        return self.state
    
    def rotate(self, action):
        '''
        Rotate the cube based on the action.
        - Action is an integer representing a move (e.g., 0-11 for 6 faces * 2 directions).
        - 0-5 for clockwise, 6-11 for counter-clockwise
        - 0: U, 1: L, 2: F, 3: R, 4: B, 5: D
        - 6: U', 7: L', 8: F', 9: R', 10: B', 11: D'
        '''

        idxs = np.array([
            # 0:Up
            [6,3,0,7,4,1,8,5,2,18,19,20,12,13,14,15,16,17,27,28,29,21,22,23,24,25,26,
                   36,37,38,30,31,32,33,34,35,9,10,11,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],
            # 1:Left
            [44,1,2,41,4,5,38,7,8,15,12,9,16,13,10,17,14,11,0,19,20,3,22,23,6,25,26,
                   27,28,29,30,31,32,33,34,35,36,37,51,39,40,48,42,43,45,18,46,47,21,49,50,24,52,53],
            # 2:Front
            [0,1,2,3,4,5,17,14,11,9,10,45,12,13,46,15,16,47,24,21,18,25,22,19,26,23,20,
                   6,28,29,7,31,32,8,34,35,36,37,38,39,40,41,42,43,44,33,30,27,48,49,50,51,52,53],
            # 3:Right
            [0,1,20,3,4,23,6,7,26,9,10,11,12,13,14,15,16,17,18,19,47,21,22,50,24,25,53,
                   33,30,27,34,31,28,35,32,29,8,37,38,5,40,41,2,43,44,45,46,42,48,49,39,51,52,36],
            # 4:Back
            [29,32,35,3,4,5,6,7,8,2,10,11,1,13,14,0,16,17,18,19,20,21,22,23,24,25,26,
                   27,28,53,30,31,52,33,34,51,42,39,36,43,40,37,44,41,38,45,46,47,48,49,50,9,12,15],
            # 5:Down
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,42,43,44,18,19,20,21,22,23,15,16,17,
                   27,28,29,30,31,32,24,25,26,36,37,38,39,40,41,33,34,35,51,48,45,52,49,46,53,50,47],
            # 6:Up'
            [2,5,8,1,4,7,0,3,6,36,37,38,12,13,14,15,16,17,9,10,11,21,22,23,24,25,26,
                   18,19,20,30,31,32,33,34,35,27,28,29,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],
            # 7:Left'
            [18,1,2,21,4,5,24,7,8,11,14,17,10,13,16,9,12,15,45,19,20,48,22,23,51,25,26,
                   27,28,29,30,31,32,33,34,35,36,37,6,39,40,3,42,43,0,44,46,47,41,49,50,38,52,53],
            # 8:Front'
            [0,1,2,3,4,5,27,30,33,9,10,8,12,13,7,15,16,6,20,23,26,19,22,25,18,21,24,
                   47,28,29,46,31,32,45,34,35,36,37,38,39,40,41,42,43,44,11,14,17,48,49,50,51,52,53],
            # 9:Right'
            [0,1,42,3,4,39,6,7,36,9,10,11,12,13,14,15,16,17,18,19,2,21,22,5,24,25,8,
                   29,32,35,28,31,34,27,30,33,53,37,38,50,40,41,47,43,44,45,46,20,48,49,23,51,52,26],
            # 10:Back'
            [15,12,9,3,4,5,6,7,8,51,10,11,52,13,14,53,16,17,18,19,20,21,22,23,24,25,26,
                   27,28,0,30,31,1,33,34,2,38,41,44,37,40,43,36,39,42,45,46,47,48,49,50,35,32,29],
            # 11:Down'
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25,26,18,19,20,21,22,23,33,34,35,
                   27,28,29,30,31,32,42,43,44,36,37,38,39,40,41,15,16,17,47,50,53,46,49,52,45,48,51]])

        new_state = self.state[idxs[action]] 
        self.state = new_state     

    def compute_similarity_reward(self, state):
        reward = 0
        for face in range(6):
            face_stickers = state[face * 9: (face + 1) * 9]
            center = face_stickers[4]
            reward += np.sum(face_stickers == center)
        return reward / 54  # Normalize to 0-1  
        
    def calculate_reward(self, prev_state, new_state):
        '''
        Calculate the reward based on the current state.
        '''
        if self.is_solved():
            return 100.0
        
        prev_score = self.compute_similarity_reward(prev_state)
        new_score = self.compute_similarity_reward(new_state)
        delta = new_score - prev_score
        return delta * 10 - 0.1  # shaping + step penalty

    def is_solved(self):
        '''
        Check if the Rubik's Cube is solved.
        '''
        return np.all(self.state == self.get_solved_cube())
    
    def render(self, mode='human'):
        '''
        Render the Rubik's Cube.
        '''

        cube_array = self.state.reshape(6, 3, 3)

        color_map = {
            0: '#FFFFFF',  # White
            1: '#FFFF00',  # Yellow
            2: '#FF0000',  # Red
            3: '#FFA500',  # Orange
            4: '#00FF00',  # Green
            5: '#0000FF'   # Blue
        }

        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot each face in its respective position
        # Top face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((3 + j, 6 - i), 1, 1, color=color_map[cube_array[0, i, j]], ec='black'))

        # Left face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, color=color_map[cube_array[1, i, j]], ec='black'))

        # Front face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((3 + j, 3 - i), 1, 1, color=color_map[cube_array[2, i, j]], ec='black'))

        # Right face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((6 + j, 3 - i), 1, 1, color=color_map[cube_array[3, i, j]], ec='black'))

        # Back face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((9 + j, 3 - i), 1, 1, color=color_map[cube_array[4, i, j]], ec='black'))

        # Bottom face
        for i in range(3):
            for j in range(3):
                ax.add_patch(plt.Rectangle((3 + j, -i), 1, 1, color=color_map[cube_array[5, i, j]], ec='black'))

        # Set limits and aspect ratio
        ax.set_xlim(0, 12)
        ax.set_ylim(-3, 9)
        ax.set_aspect('equal')
        ax.axis('off')

        # Show the plot
        plt.show()    

