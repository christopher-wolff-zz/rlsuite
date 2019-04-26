import os
import itertools
import time

import numpy as np

from rlsuite.utils import Logger


def qlearning(
    env_fn,
    alpha=0.1,
    epsilon=0.1,
    gamma=0.99,
    num_episodes=1000,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """Off-policy TD control.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        alpha (float): The step size.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): The directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        A tuple containing the final Q table and policy.

    """
    # --- Parameter validation ---
    assert alpha > 0 and epsilon <= 1, 'epsilon must be in (0, 1]'
    assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in [0, 1]'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_episodes > 0, 'num_episodes must be positive'

    # --- Initialization ---
    train_logger = Logger(output_dir=os.path.join(data_dir), output_fname='train_statistics.tsv', log_freq=log_freq)
    eval_logger = Logger(output_dir=os.path.join(data_dir), output_fname='eval_statistics.tsv', log_freq=log_freq)

    env = env_fn()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((num_states, num_actions))
    behavior_policy = np.full((num_states, num_actions), 1 / num_actions)
    target_policy = np.full((num_states, num_actions), 1 / num_actions)

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_episodes):
        # Simulate one episode using behavior policy
        train_episode_length = 0
        train_episode_return = 0
        state = env.reset()
        done = False
        while not done:
            # Choose action from behavior policy
            action = np.random.choice(num_actions, p=behavior_policy[state])

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # Update Q for the current state
            target = reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])

            # Update policy for the current state
            best_actions = np.where(Q[state] == Q[state].max())[0]
            best_action = np.random.choice(best_actions)
            for a in np.arange(num_actions):
                if a == best_action:
                    behavior_policy[state, a] = 1 - epsilon + epsilon / num_actions
                    target_policy[state, a] = 1
                else:
                    behavior_policy[state, a] = epsilon / num_actions
                    target_policy[state, a] = 0

            # Update state and statistics
            state = next_state
            train_episode_length += 1
            train_episode_return += reward

        train_logger.store(
            iteration=i,
            episode_length=train_episode_length,
            episode_return=train_episode_return,
            time=time.time()-start_time,
        )

        # Simulate one episode using target policy
        eval_episode_length = 0
        eval_episode_return = 0
        state = env.reset()
        done = False
        while not done:
            # Choose action from target policy
            action = np.random.choice(num_actions, p=target_policy[state])

            # Take action in the environment
            state, reward, done, _ = env.step(action)

            # Update state and statistics
            eval_episode_length += 1
            eval_episode_return += reward

        # Log statistics
        eval_logger.store(
            iteration=i,
            episode_length=eval_episode_length,
            episode_return=eval_episode_return,
            time=time.time()-start_time,
        )

    # --- Deinitialization ---
    env.close()
    return Q, target_policy


if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/tmp/experiments/qlearning')
    parser.add_argument('--log_freq', type=int, default=0)
    args = parser.parse_args()

    qlearning(
        env_fn=lambda: gym.make(args.env),
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        num_episodes=args.num_episodes,
        seed=args.seed,
        data_dir=args.data_dir,
        log_freq=args.log_freq,
    )
