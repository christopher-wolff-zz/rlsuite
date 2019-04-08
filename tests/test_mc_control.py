import gym

from rlsuite.algos import mc_control
from rlsuite.envs import gridworld


def test_first_visit():
    mc_control(
        env_fn=lambda: gym.make('GridWorld-v0'),
        method='first_visit',
        seed=0,
        data_dir='/tmp/tests/mc_control',
    )


def test_every_visit():
    mc_control(
        env_fn=lambda: gym.make('GridWorld-v0'),
        method='every_visit',
        seed=0,
        data_dir='/tmp/tests/mc_control',
    )
