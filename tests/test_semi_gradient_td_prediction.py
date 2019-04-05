import gym
import tensorflow as tf
from tensorflow.keras import layers

from rlsuite.algos import semi_gradient_td_prediction


def test_semi_gradient_td_prediction():
    semi_gradient_td_prediction(
        env_fn=lambda: gym.make('CartPole-v1'),
        policy=lambda *args, **kwargs: [0.5, 0.5],
        value_fn=tf.keras.Sequential([
            layers.Dense(10, input_shape=(4,), activation='relu'),
            layers.Dense(1),
        ]),
        alpha=0.01,
        gamma=0.99,
        num_episodes=100,
        seed=0,
        data_dir='/tmp/tests/semi_gradient_td_prediction',
        log_freq=1,
    )
