import gym
import tensorflow as tf
from tensorflow.keras import layers

from rlsuite.algos import reinforce


def test_reinforce():
    reinforce(
        env_fn=lambda: gym.make('CartPole-v1'),
        policy=tf.keras.Sequential([
            layers.Dense(5, activation='relu'),
            layers.Dense(2, input_shape=(4,), activation='softmax'),
        ]),
        num_episodes=10000,
        data_dir='/tmp/tests/reinforce',
        log_freq=10,
    )
