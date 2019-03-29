import itertools
import time

import numpy as np

from rlsuite import utils


def mc_control(
    env_fn,
    epsilon=0.1,
    gamma=0.99,
    num_episodes=1000,
    method='first_visit',
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """On-policy Monte Carlo control.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        method (str): Either 'first_visit' or 'every_visit'.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): Optional. A directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        A tuple containing the final Q table and policy.

    """
    # --- Parameter validation ---
    assert epsilon >= 0 and epsilon <= 1, 'epsilon must be in [0, 1]'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_episodes > 0, 'num_episodes must be positive'
    assert method in ['first_visit', 'every_visit'], "method must be 'first_visit' or 'every_visit'"

    # --- Initialization ---
    logger = utils.Logger(output_dir=data_dir, log_freq=log_freq)
    logger.log_params(
        epsilon=epsilon,
        gamma=gamma,
        num_episodes=num_episodes,
        method=method,
        seed=seed,
        data_dir=data_dir,
    )

    env = env_fn()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((num_states, num_actions))
    pi = np.full((num_states, num_actions), 1 / num_actions)

    # A 2d array of lists that holds the returns seen from each state-action pair
    returns = np.empty((num_states, num_actions), dtype=object)
    returns[...] = [[list() for _ in range(num_actions)] for _ in range(num_states)]

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_episodes):
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
            if not visited[state, action] or method == 'every_visit':
                # Update Q table
                Q[state, action] = np.mean(returns[state, action])
                # Improve policy
                best_actions = np.where(Q[state] == Q[state].max())[0]
                best_action = np.random.choice(best_actions)
                for a in np.arange(num_actions):
                    if a == best_action:
                        pi[state, a] = 1 - epsilon + epsilon / num_actions
                    else:
                        pi[state, a] = epsilon / num_actions
            visited[state, action] = True

        logger.log_stats(
            iteration=i,
            episode_length=len(episode),
            episode_return=episode_return,
            time=time.time()-start_time,
        )

    # --- Deinitialization ---
    env.close()
    return Q, pi


if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--method', type=str, default='first_visit')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/tmp/experiments/mc_control')
    parser.add_argument('--log_freq', type=int, default=0)
    args = parser.parse_args()

    mc_control(
        env_fn=lambda: gym.make(args.env),
        epsilon=args.epsilon,
        gamma=args.gamma,
        method=args.method,
        num_episodes=args.num_episodes,
        seed=args.seed,
        data_dir=args.data_dir,
        log_freq=args.log_freq,
    )
