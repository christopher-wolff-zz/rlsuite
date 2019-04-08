import gym
import tensorflow as tf
from tensorflow.keras import layers

from rlsuite.algos import td_lambda


def test_td_lambda():
    td_lambda(
        env_fn=lambda: gym.make('CartPole-v1'),
        policy=lambda *args, **kwargs: [0.5, 0.5],
        value_fn=tf.keras.Sequential([
            layers.Dense(10, input_shape=(4,), activation='relu'),
            layers.Dense(1),
        ]),
        num_episodes=100,
        data_dir='/tmp/tests/td_lambda',
    )
