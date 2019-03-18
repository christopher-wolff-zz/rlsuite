import gym

from rlsuite.algos import mc_control
from rlsuite.envs import gridworld


def test_first_visit():
    mc_control(
        env_fn=lambda: gym.make('GridWorld-v0'),
        epsilon=0.5,
        gamma=0.9,
        method='first_visit',
        num_episodes=10,
        seed=0,
        data_dir='/tmp/tests/mc_control'
    )


def test_every_visit():
    mc_control(
        env_fn=lambda: gym.make('GridWorld-v0'),
        epsilon=0.5,
        gamma=0.9,
        method='every_visit',
        num_episodes=10,
        seed=0,
        data_dir='/tmp/tests/mc_control'
    )
