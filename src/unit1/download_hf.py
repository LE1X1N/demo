import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

repo_id = "LE1X1N/ppo-LunarLander-v3"
filename = "ppo-LunarLander-v3.zip"

custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _:0.0,
    "clip_range": lambda _: 0.0,
}

# load
checkpoint = load_from_hub(repo_id, filename)
model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)

# eval
eval_env = Monitor(gym.make("LunarLander-v3"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")