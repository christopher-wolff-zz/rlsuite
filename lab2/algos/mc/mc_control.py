import itertools
import os
import sys

import numpy as np
import tensorflow as tf


def mc_control(
    env_fn,
    epsilon,
    gamma,
    num_episodes,
    method='first_visit',
    base_dir='/tmp/experiments',
    exp_name='mc_control',
    seed=0
):
    """On-policy Monte Carlo control.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        method (str): Either 'first_visit' or 'every_visit'.
        base_dir (str): The base directory for storing experiment data.
        exp_name (str): The name of the experiment.
        seed (int): A seed that fixes all randomness if possible.

    """
    # --- Parameter validation ---
    assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in [0, 1]'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_episodes > 0, 'num_episodes must be positive'
    assert method in ['first_visit', 'every_visit'], "method must be 'first_visit' or 'every_visit'"

    # --- Logging ---
    # TODO

    # --- Initialization ---
    # Summary writer
    summary_writer = tf.summary.FileWriter(os.path.join(base_dir, exp_name))

    # Environment
    env = env_fn()
    env.seed(seed)

    num_states = env.nS
    num_actions = env.nA

    # Policy - pi[s] is a vector of probabilities for each action in state s.
    pi = np.full((num_states, num_actions), 1 / num_actions)

    # Q-table
    Q = np.zeros((num_states, num_actions))

    # Return buffer
    returns = np.empty((num_states, num_actions), dtype=object)
    returns[...] = [[list() for _ in range(num_actions)] for _ in range(num_states)]

    # --- Main loop ---
    for i in range(num_episodes):
        # Console logging
        sys.stdout.write(f'Episode {i}/{num_episodes}\r')
        sys.stdout.flush()

        # Initialize episode statistics
        episode_return = 0

        # Roll out an episode
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(num_actions, p=pi[state])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            episode_return += reward

        # Update Q-table and policy
        G = 0
        visited = np.full((num_states, num_actions), False)
        for (state, action, reward) in reversed(episode):
            G = gamma * G + reward
            returns[state, action].append(G)
            if not visited[state, action] or method == 'every-visit':
                Q[state, action] = np.mean(returns[state, action])
                best_action = np.argmax(Q[state])
                for a in np.arange(num_actions):
                    if a == best_action:
                        pi[state, a] = 1 - epsilon + epsilon / num_actions
                    else:
                        pi[state, a] = epsilon / num_actions
            visited[state, action] = True

        # Write episode summary
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='episode_length', simple_value=len(episode)),
            tf.Summary.Value(tag='episode_return', simple_value=episode_return)
        ])
        summary_writer.add_summary(summary, global_step=i)

    # --- Deinitialization ---
    env.close()
    summary_writer.close()


if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--method', type=str, default='first_visit')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--base_dir', type=str, default='/tmp/experiments')
    parser.add_argument('--exp_name', type=str, default='mc_control')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    # TODO: handle custom environments correctly
    gym_envs = [env.id for env in gym.envs.registry.all()]
    if args.env not in gym_envs:
        import lab2.envs.gridworld
        env_fn = lambda: gym.make('GridWorld-v0')
    else:
        env_fn = lambda: gym.make(args.env)

    mc_control(
        env_fn=env_fn,
        epsilon=args.epsilon,
        gamma=args.gamma,
        method=args.method,
        num_episodes=args.num_episodes,
        base_dir=args.base_dir,
        exp_name=args.exp_name,
        seed=args.seed
    )
