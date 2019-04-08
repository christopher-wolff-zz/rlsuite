import gym

from rlsuite.algos import expected_sarsa
from rlsuite.envs import gridworld


def test_expected_sarsa():
    expected_sarsa(
        env_fn=lambda: gym.make('GridWorld-v0'),
        data_dir='/tmp/tests/expected_sarsa',
    )
