import time

import numpy as np

from rlsuite import utils


def lstd(
    env_fn,
    policy,
    epsilon=0.01,
    gamma=0.99,
    num_iter=1000,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """Least squares temporal difference learning.

    Args:
        env_fn (callable): A function that creates an instance of an environment.
        policy (callable): A function mapping from state space to a probability
            over actions.
        epsilon (float): A small positive constant for numeric stability.
        gamma (float): The discount factor.
        num_iter (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): Optional. A directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        The parameters of the approximate value function under the given policy.

    """
    # --- Parameter validation ---
    assert epsilon > 0, 'epsilon must be positive'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_iter > 0, 'num_iter must be positive'

    # --- Initialization ---
    logger = utils.Logger(output_dir=data_dir, log_freq=log_freq)

    env = env_fn()
    num_actions = env.action_space.n
    num_features = env.observation_space.shape[0]

    env.seed(seed)
    np.random.seed(seed)

    A_inv = 1 / epsilon * np.identity(num_features)
    b = np.zeros(num_features)
    w = np.zeros(num_features)

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_iter):
        # Initialize statistics
        episode_length = 0
        episode_return = 0

        # Reset
        x = env.reset()
        done = False

        # Simulate one episode
        while not done:
            action = np.random.choice(num_actions, p=policy(x))
            next_x, reward, done, _ = env.step(action)

            v = A_inv.transpose().dot(x - gamma * next_x)
            A_inv -= np.outer(A_inv.dot(x), v.transpose()) / (1 + np.inner(v, x))
            b += reward * x
            w = A_inv.dot(b)

            # Update feature vector and statistics
            x = next_x
            episode_length += 1
            episode_return += reward

        logger.log_stats(
            iteration=i,
            episode_length=episode_length,
            episode_return=episode_return,
            time=time.time()-start_time,
        )

    return w
