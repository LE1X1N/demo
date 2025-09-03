from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import numpy as np
import tqdm

# import pickle5 as pickle  # not support for python >= 3.8
from tqdm import tqdm

"""
    initialzie q-table
"""
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


"""
    Greedy policy
    - Epsilon-greedy policy (action policy)
    - Greedy-policy (updating policy)
"""
def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = np.random.rand(1)
    if random_num < epsilon:
        # random
        action = np.random.choice(Qtable.shape[1])
    else:
        # greedy
        action = np.argmax(Qtable[state])
    return action



"""
    Training Loop
"""
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable, learning_rate, gamma):
    
    epsilon = max_epsilon

    for episode in tqdm(range(n_training_episodes)):
        epsilon = max(min_epsilon, epsilon * (1 - decay_rate))  # decay epsilon
        state, info = env.reset()
        terminated = False
        truncated = False

        for step in range(max_steps):
            # updating policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            # observe R_t+1, S_t+1
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # IMPORTANT!!! update Q-table
            # Q(S_t, A_t) <- Q(S_t, A_t) + \alpha (R_t+1 + \gemma * max_a(Q_t+1, a) - Q(S_t, A_t))
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            # terminated
            if terminated:
                break
            state = new_state
    
    return Qtable


"""
    Evaluation method
"""
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    episode_rewards = []

    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # acting policy
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)
    
    # mean and std
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward
