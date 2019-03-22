import gym

from rlsuite.algos import sarsa
from rlsuite.envs import gridworld


def test_sarsa():
    sarsa(
        env_fn=lambda: gym.make('GridWorld-v0'),
        alpha=0.1,
        epsilon=0.1,
        gamma=0.99,
        num_episodes=1000,
        seed=0,
        data_dir='/tmp/tests/sarsa',
        log_freq=0,
    )
