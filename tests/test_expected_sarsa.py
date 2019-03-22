import gym

from rlsuite.algos import expected_sarsa
from rlsuite.envs import gridworld


def test_expected_sarsa():
    expected_sarsa(
        env_fn=lambda: gym.make('GridWorld-v0'),
        alpha=0.1,
        epsilon=0.1,
        gamma=0.99,
        num_episodes=1000,
        seed=0,
        data_dir='/tmp/tests/expected_sarsa',
        log_freq=0,
    )
