import gymnasium as gym
from huggingface_sb3 import load_from_hub,  package_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

"""
Lunarlander-v3
"""
env = gym.make("LunarLander-v3")
observation, info = env.reset()

print(f"Initial State S_0 {observation}")
print(f"Initial Info: {info}")

for _ in range(20):
    # take random action
    print("**********",_ , "**********")
    action = env.action_space.sample()
    print("Action taken: ", action)
    
    # do this random action and get the observation / state
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Next state S_t+1: {observation}")
    print(f"Reward R_t+1: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}") 

    # end game
    if terminated or truncated:
        print("Environment is reset")
        observation, info = env.reset()
    
env.close()