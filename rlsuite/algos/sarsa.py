import itertools
import time

import numpy as np

from rlsuite import utils


def sarsa(
    env_fn,
    alpha=0.1,
    epsilon=0.1,
    gamma=0.99,
    num_episodes=1000,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """On-policy TD control.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        alpha (float): The step size.
        epsilon (float): The exploration rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): Optional. A directory for storing experiment data.
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
    logger = utils.Logger(output_dir=data_dir, log_freq=log_freq)
    logger.log_params(
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma,
        num_episodes=num_episodes,
        seed=seed,
        data_dir=data_dir,
        log_freq=log_freq,
    )

    env = env_fn()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    Q = np.zeros((num_states, num_actions))
    pi = np.full((num_states, num_actions), 1 / num_actions)

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_episodes):
        # Initialize episode statistics
        episode_length = 0
        episode_return = 0

        # Simulate one episode
        state = env.reset()
        action = np.random.choice(num_actions, p=pi[state])
        done = False
        while not done:
            # Take action and observe next state
            next_state, reward, done, _ = env.step(action)

            # Determine next action and next state
            next_action = np.random.choice(num_actions, p=pi[state])

            # Update Q for the current state
            target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])

            # Update policy for the current state
            best_actions = np.where(Q[state] == Q[state].max())[0]
            best_action = np.random.choice(best_actions)
            for a in np.arange(num_actions):
                if a == best_action:
                    pi[state, a] = 1 - epsilon + epsilon / num_actions
                else:
                    pi[state, a] = epsilon / num_actions

            # Update state and action
            state = next_state
            action = next_action

            # Update statistics
            episode_length += 1
            episode_return += reward

        logger.log_stats(
            iteration=i,
            episode_length=episode_length,
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
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/tmp/experiments/sarsa')
    parser.add_argument('--log_freq', type=int, default=0)
    args = parser.parse_args()

    sarsa(
        env_fn=lambda: gym.make(args.env),
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        num_episodes=args.num_episodes,
        seed=args.seed,
        data_dir=args.data_dir,
        log_freq=args.log_freq,
    )
