import itertools

import numpy as np

from rlsuite.utils import Logger


def policy_eval(
    env_fn,
    policy,
    gamma,
    theta,
    max_iter=0,
    seed=0,
    data_dir=None,
    log_freq=1
):
    """Iterative policy evaluation for tabular environments.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        policy (np.array): An array of size `num_states` by `num_actions` where
            the elements represent action probabilities under the target policy.
        gamma (float): The discount factor.
        theta (float): A small threshold determining the accuracy of the
            estimation. The algorithms terminates if the largest change to any
            state's value is lower than this threshold.
        max_iter (int): The maximum number of iterations. If non-positive, the
            only termination criterion will be `theta`.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): A directory for storing experiment results.
        log_freq (int): The interval for logging to the console.

    Returns:
        A 1-d `np.array` representing the resulting value function.

    """
    # --- Parameter validation ---
    assert theta > 0, 'theta must be greater than 0'

    # --- Initialization ---
    # Logger
    logger = Logger(output_dir=data_dir)
    logger.log_params(gamma=gamma, theta=theta, max_iter=max_iter)

    # Environment
    env = env_fn()
    env.seed(seed)

    # State and action space
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Value function
    V = np.zeros((num_states))

    # --- Main loop ---
    for i in itertools.count():
        delta = 0
        for state in range(num_states):
            old_v = V[state]
            new_v = 0
            for action in env.P[state]:
                action_value = 0
                for trans_prob, next_state, reward, _ in env.P[state][action]:
                    action_value += trans_prob * (reward + gamma * V[next_state])
                new_v += policy[state][action] * action_value
            V[state] = new_v
            delta = max(delta, abs(new_v - old_v))

        logger.log_stats(step=i, delta=delta)

        if delta < theta or (max_iter > 0 and i > max_iter):
            break

    # --- Deinitialization ---
    env.close()
    return V
