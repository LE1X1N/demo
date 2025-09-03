import gymnasium as gym
import numpy as np

from q_learning import initialize_q_table, train, evaluate_agent


def run_demo(env_id, max_steps, n_training_episodes, n_eval_episodes, eval_seed, learning_rate, gamma, max_epsilon, min_epsilon, decay_rate):
    """
    Frozen Lake
        Navigate from starting state (S) to the goal state (G) by walking on the frozen tiles (F) 
        and avoid holes (H)

        rewards:
            Reach goal: +1
            Reach hole: 0
            Reach frozen: 0

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
    state_space = env.observation_space.n
    action_space = env.action_space.n
    print(f"There are {state_space} possible states")
    print(f"There are {action_space} possible actions")

    Qtable_frozenlake = initialize_q_table(state_space, action_space)

    # training the Q-table 
    # train from scratch or load from checkpoint
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake, learning_rate, gamma)
    # Qtable_frozenlake = np.load("src/unit2/ckpt/Qtable_frozenlake.npy")

    # evaluation
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
    print(f"Mean reward={mean_reward} +/- {std_reward}")
    
    return Qtable_frozenlake, env


if __name__ == "__main__":

    """
        Hyper-parameters
    """
    # training parameters
    n_training_episodes = 10000
    learning_rate = 0.7

    # evaluation parameters
    n_eval_episodes = 100

    # environment parameters
    env_id = "FrozenLake-v1"
    max_steps = 99
    gamma = 0.95
    eval_seed = []

    # exploration parameters
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005

    run_demo(env_id, max_steps, n_training_episodes, n_eval_episodes, eval_seed, learning_rate, gamma, max_epsilon, min_epsilon, decay_rate)