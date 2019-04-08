import gym
import tensorflow as tf
from tensorflow.keras import layers

from rlsuite.algos import gradient_mc_prediction


def test_gradient_mc_prediction():
    gradient_mc_prediction(
        env_fn=lambda: gym.make('CartPole-v1'),
        policy=lambda *args, **kwargs: [0.5, 0.5],
        value_fn=tf.keras.Sequential([
            layers.Dense(10, input_shape=(4,), activation='relu'),
            layers.Dense(1),
        ]),
        data_dir='/tmp/tests/gradient_mc_prediction',
    )
