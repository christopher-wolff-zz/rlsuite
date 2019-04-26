from gym.envs.registration import register

register(
    id='RandomWalk-v0',
    entry_point='rlsuite.envs.random_walk.random_walk:RandomWalk',
    max_episode_steps=50,
)
