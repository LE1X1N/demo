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

print("_____OBSERVATION SPACE______\n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())
