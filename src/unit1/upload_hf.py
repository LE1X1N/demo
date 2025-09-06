import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from huggingface_sb3 import package_to_hub

# repo_id and env_id
repo_id = "LE1X1N/ppo-LunarLander-v3"
env_id = "LunarLander-v3"

# pre-trained model
model_name = "ppo-LunarLander-v3"
model = PPO.load(f"src/unit1/ckpt/{model_name}.zip")

# evalucation env
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

model_architexture = "PPO"
commit_message = "policy for LunarLander-v3 using PPO algorithm"

# method save, evaluate, generate a model card
package_to_hub(model=model,
               model_name=model_name,
               model_architecture=model_architexture,
               env_id=env_id,
               eval_env=eval_env,
               repo_id=repo_id,
               commit_message=commit_message)
