import time

import numpy as np
import tensorflow as tf

from rlsuite import utils


def td_lambda(
    env_fn,
    policy,
    value_fn,
    alpha=0.01,
    lambda_=0.99,
    gamma=0.99,
    num_episodes=1000,
    seed=0,
    data_dir=None,
    log_freq=0,
):
    """Semi-gradient TD(lambda) for estimating the value of a given policy.

    Args:
        env_fn (callable): A function that creates an instance of an environment.
        policy (callable): The policy to be evaluated, maps from the observation
            space to a probability over actions.
        value_fn (tf.keras.Model): The value function model.
        alpha (float): The learning rate.
        lambda_ (float): The trace decay rate.
        gamma (float): The discount factor.
        num_episodes (int): The number of episodes to run.
        seed (int): A seed that fixes all randomness if possible.
        data_dir (str): Optional. A directory for storing experiment data.
        log_freq (int): The interval for logging to the console.

    Returns:
        The approximate optimal value function. This is a reference to the same
        object that was provided as the parameter `value_fn`.

    """
    # --- Parameter validation ---
    assert alpha >= 0, 'alpha must be non-negative'
    assert lambda_ >= 0 and lambda_ <= 1, 'lambda must be in [0, 1]'
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
        x = env.reset().reshape(1, -1)  # must be in batch-form
        done = False
        z = []
        for w in value_fn.trainable_weights:
            z.append(tf.Variable(tf.zeros_like(w)))

        # Simulate one episode
        while not done:
            # Choose action and observe reward and next state
            action = np.random.choice(num_actions, p=policy(x))
            next_x, reward, done, _ = env.step(action)
            next_x = next_x.reshape(1, -1)  # must be in batch-form

            # Compute gradient of value function
            weights = value_fn.trainable_weights
            with tf.GradientTape() as tape:
                value = value_fn(x)
            grads = tape.gradient(value, weights)

            # Update `z`
            for z_, g in zip(z, grads):
                z_.assign(gamma * lambda_ * z_ + g)

            # Compute TD error
            if done:
                target = reward
            else:
                target = reward + gamma * value_fn(next_x)
            delta = target - value_fn(x)

            # Update weights
            for w, z_ in zip(weights, z):
                dw = tf.reshape(alpha * delta * z_, w.shape)
                w.assign_add(dw)

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

    # --- Deinitialization ---
    env.close()
    return value_fn
