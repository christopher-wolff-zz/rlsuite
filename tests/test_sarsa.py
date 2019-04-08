import gym

from rlsuite.algos import sarsa
from rlsuite.envs import gridworld


def test_sarsa():
    sarsa(
        env_fn=lambda: gym.make('GridWorld-v0'),
        data_dir='/tmp/tests/sarsa',
    )
