import gym

from rlsuite.algos import lstd


def test_lstd():
    lstd(
        env_fn=lambda: gym.make('CartPole-v1'),
        policy=lambda *args, **kwargs: [0.5, 0.5],
        data_dir='/tmp/tests/lstd',
    )
