import gym
import numpy as np

from rlsuite.algos import policy_eval
from rlsuite.envs import gridworld


def test_policy_eval():
    random_policy = np.full(shape=(16, 4), fill_value=0.25)
    policy_eval(
        env_fn=lambda: gym.make('GridWorld-v0'),
        policy=random_policy,
        gamma=0.99,
        theta=0.001,
        data_dir='/tmp/tests/policy_eval',
        log_freq=1,
    )
