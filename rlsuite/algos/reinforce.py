from collections import namedtuple
import time

import numpy as np
import tensorflow as tf

from rlsuite import utils


Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done'])


def reinforce(
    env_fn,
    policy,
    alpha=0.001,
    gamma=0.99,
    num_episodes=1000,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """Monte-Carlo Policy-Gradient Control.

    Args:
        env_fn (callable): A function that creates an instance of an environment.
        policy (tf.keras.Model): The policy architecture, mapping from
            observation space to a vector of action probabilities.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): Optional. A directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        A locally optimal policy.

    """
    # --- Parameter validation ---
    assert alpha >= 0, 'alpha must be non-negative'
    assert gamma >= 0 and gamma <= 1, 'gamma must be in [0, 1]'
    assert num_episodes > 0, 'num_episodes must be positive'

    # --- Initialization ---
    logger = utils.Logger(output_dir=data_dir, log_freq=log_freq)

    env = env_fn()
    num_actions = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    # --- Main loop ---
    for i in range(num_episodes):
        # Initialize episode statistics
        episode_length = 0
        episode_return = 0

        # Reset
        state = env.reset()
        done = False

        # Simulate one episode
        episode = []
        while not done:
            # Choose action and observe reward and next state
            action_probs = np.squeeze(policy(state.reshape(1, -1)))
            action = np.random.choice(num_actions, p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append(Transition(state, action, next_state, reward, done))

            # Update state and episode statistics
            state = next_state
            episode_length += 1
            episode_return += reward

        for t, transition in enumerate(episode):
            # Compute return from time t
            G = sum(gamma ** i * tr.reward for i, tr in enumerate(episode[t:]))

            # Compute gradient of policy function
            weights = policy.trainable_weights
            with tf.GradientTape() as tape:
                action_probs = tf.squeeze(policy(state.reshape(1, -1)))
                chosen_action = action_probs[transition.action]
                log_prob = tf.math.log(chosen_action)
            grads = tape.gradient(log_prob, weights)

            # Update weights
            for w, g in zip(weights, grads):
                dw = tf.reshape(alpha * gamma ** t * G * g, w.shape)
                w.assign_add(dw)

        logger.store(
            iteration=i,
            episode_length=episode_length,
            episode_return=episode_return,
            time=time.time()-start_time,
        )

    # --- Deinitialization ---
    env.close()
    return policy
