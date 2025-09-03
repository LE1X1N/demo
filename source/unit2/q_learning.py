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