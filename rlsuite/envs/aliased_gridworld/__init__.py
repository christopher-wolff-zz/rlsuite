from gym.envs.registration import register

register(
    id='AliasedGridWorld-v0',
    entry_point='rlsuite.envs.aliased_gridworld.aliased_gridworld:AliasedGridWorld',
)
