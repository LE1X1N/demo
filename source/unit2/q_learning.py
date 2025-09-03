from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm

# import pickle5 as pickle  # not support for python >= 3.8
from tqdm.notebook import tqdm

"""
Frozen Lake
    Navigate from starting state (S) to the goal state (G) by walking on the frozen tiles (F) 
    and avoid holes (H)

    one desc could be :
    
    "4x4":[
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ]
"""

# create a FrozenLake-v1 environment  
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array")

print("\n_____OBSERVATION SPACE______")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())

print("\n_____ACTION SPACE_____")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())

# initialzie q-table
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

state_space = env.observation_space.n
action_space = env.action_space.n
print(f"There are {state_space} possible states")
print(f"There are {action_space} possible actions")

Qtable_frozenlake = initialize_q_table(state_space, action_space)
