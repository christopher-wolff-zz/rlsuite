import itertools
import time

import numpy as np

from rlsuite.utils import Logger


def nstep_prediction(
    env_fn,
    policy,
    alpha,
    n,
    gamma,
    num_iter,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """n-step value prediction.

    Args:
        env_fn (func): A function that creates an instance of an environment.
        policy (np.array): An array of size `num_states` by `num_actions` where
            the elements represent action probabilities under the policy to be
            evaluated.
        alpha (float): The step size.
        n (int): The number of steps determining the depth of each TD update.
        gamma (float): The discount factor.
        num_iter (int): The number of iterations to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): The directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        A 1-d `np.array` representing the resulting value function.

    """
    # --- Parameter validation ---
    assert alpha > 0 and epsilon <= 1, 'epsilon must be in (0, 1]'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_iter > 0, 'num_iter must be positive'

    # --- Initialization ---
    logger = Logger(output_dir=data_dir, log_freq=log_freq)

    env = env_fn()
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    V = np.zeros((num_states))

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_episodes):
        # Initialize episode statistics
        episode_length = 0
        episode_return = 0

        # Simulate one episode
        state = env.reset()
        done = False
        while not done:
            # Choose action from current policy
            action = np.random.choice(num_actions, p=pi[state])

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
                    pi[state, a] = 1 - epsilon + epsilon / num_actions
                else:
                    pi[state, a] = epsilon / num_actions

            # Update state
            state = next_state

            # Update statistics
            episode_length += 1
            episode_return += reward

        # Log statistics
        logger.store(
            iteration=i,
            episode_length=episode_length,
            episode_return=episode_return,
            time=time.time()-start_time
        )

    # --- Deinitialization ---
    env.close()
