import gym

from rlsuite.algos import qlearning
from rlsuite.envs import gridworld


def test_qlearning():
    qlearning(
        env_fn=lambda: gym.make('GridWorld-v0'),
        data_dir='/tmp/tests/qlearning',
    )
