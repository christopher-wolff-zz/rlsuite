import itertools
import logging
import os
import sys

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sarsa(
    logdir,
    env_fn,
    alpha,
    epsilon,
    gamma,
    num_episodes,
    exp_name='expected_sarsa',
    seed=0
):
    """On-policy TD control.

    Args:
        logdir (str): The base directory for storing experiment data.
        env_fn (func): A function that creates an instance of an environment.
        alpha (float): The step size.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.

    """
    # --- Parameter validation ---
    assert alpha > 0 and epsilon <= 1, 'epsilon must be in (0, 1]'
    assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in [0, 1]'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_episodes > 0, 'num_episodes must be positive'

    # --- Parameter logging ---
    logger.info(f'ARG logdir {logdir}')
    logger.info(f'ARG alpha {alpha}')
    logger.info(f'ARG epsilon {epsilon}')
    logger.info(f'ARG gamma {gamma}')
    logger.info(f'ARG num_episodes {num_episodes}')
    logger.info(f'ARG seed {seed}')

    # --- Initialization ---
    # Summary writer
    summary_writer = tf.summary.FileWriter(logdir)

    # Environment
    env = env_fn()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Seeds
    env.seed(seed)
    np.random.seed(seed)

    # Policy - pi[s] is a vector of probabilities for each action in state s.
    pi = np.full((num_states, num_actions), 1 / num_actions)

    # Q-table
    Q = np.zeros((num_states, num_actions))

    # --- Main loop ---
    for i in range(num_episodes):
        # Console logging
        sys.stdout.write(f'Episode {i}/{num_episodes}\r')
        sys.stdout.flush()

        # Initialize episode statistics
        episode_length = 0
        episode_return = 0

        # Simulate one episode
        state = env.reset()
        done = False
        while not done:
            # Take action and observe next state
            action = np.random.choice(num_actions, p=pi[state])
            next_state, reward, done, _ = env.step(action)

            # Update Q for the current state
            expected_value = sum(pi[next_state, a] * Q[next_state, a]
                                 for a in np.arange(num_actions))
            target = reward + gamma * expected_value
            Q[state, action] += alpha * (target - Q[state, action])

            # Update policy for the current state
            best_actions = np.where(Q[state] == Q[state].max())[0]
            best_action = np.random.choice(best_actions)
            for a in np.arange(num_actions):
                if a == best_action:
                    pi[state, a] = 1 - epsilon + epsilon / num_actions
                else:
                    pi[state, a] = epsilon / num_actions

            # Update state
            state = next_state

            # Update statistics
            episode_length += 1
            episode_return += reward

        # Write episode summary
        with summary_writer.as_default():
            tf.summary.scalar('episode_length', episode_length, step=i)
            tf.summary.scalar('episode_return', episode_return, step=i)

    # --- Deinitialization ---
    env.close()
    summary_writer.close()


if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='/tmp/exp/expected_sarsa')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    sarsa(
        env_fn=lambda: gym.make(args.env),
        logdir=args.logdir,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        num_episodes=args.num_episodes,
        seed=args.seed
    )
