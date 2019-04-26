import io
import sys

from gym import Env, spaces
from gym.utils import seeding
import numpy as np


# States
TOP_LEFT = 0
TOP_MID = 1
TOP_RIGHT = 2
BOT_LEFT = 3
BOT_MID = 4
BOT_RIGHT = 5
LEFT_GRAY = 6
RIGHT_GRAY = 7

# Actions
LEFT = 0
RIGHT = 1
DOWN = 2


def categorical_sample(prob_n, np_random):
    """Sample from a categorical distribution."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class AliasedGridWorld(Env):

    def __init__(self):
        self.P = {
            TOP_LEFT: {
                LEFT: [(1.0, TOP_LEFT, 0.0, False)],
                RIGHT: [(1.0, LEFT_GRAY, 0.0, False)],
                DOWN: [(1.0, BOT_LEFT, 0.0, False)],
            },
            LEFT_GRAY: {
                LEFT: [(1.0, TOP_LEFT, 0.0, False)],
                RIGHT: [(1.0, TOP_MID, 0.0, False)],
                DOWN: [(1.0, LEFT_GRAY, 0.0, False)],
            },
            TOP_MID: {
                LEFT: [(1.0, LEFT_GRAY, 0.0, False)],
                RIGHT: [(1.0, RIGHT_GRAY, 0.0, False)],
                DOWN: [(1.0, BOT_MID, 0.0, False)],
            },
            RIGHT_GRAY: {
                LEFT: [(1.0, TOP_MID, 0.0, False)],
                RIGHT: [(1.0, TOP_RIGHT, 0.0, False)],
                DOWN: [(1.0, RIGHT_GRAY, 0.0, False)],
            },
            TOP_RIGHT: {
                LEFT: [(1.0, RIGHT_GRAY, 0.0, False)],
                RIGHT: [(1.0, TOP_RIGHT, 0.0, False)],
                DOWN: [(1.0, BOT_RIGHT, 0.0, False)],
            },
            BOT_LEFT: {
                LEFT: [(1.0, BOT_LEFT, -1.0, True)],
                RIGHT: [(1.0, BOT_LEFT, -1.0, True)],
                DOWN: [(1.0, BOT_LEFT, -1.0, True)],
            },
            BOT_MID: {
                LEFT: [(1.0, BOT_MID, 1.0, True)],
                RIGHT: [(1.0, BOT_MID, 1.0, True)],
                DOWN: [(1.0, BOT_MID, 1.0, True)],
            },
            BOT_RIGHT: {
                LEFT: [(1.0, BOT_RIGHT, -1.0, True)],
                RIGHT: [(1.0, BOT_RIGHT, -1.0, True)],
                DOWN: [(1.0, BOT_RIGHT, -1.0, True)],
            },
        }
        self.isd = np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5])
        self.nS = 7  # not 8
        self.nA = 3

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.np_random = None
        self.seed()

        self.state = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = categorical_sample(self.isd, self.np_random)
        return self._get_obs()

    def step(self, action):
        transitions = self.P[self.state][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, next_state, reward, done = transitions[i]
        self.state = next_state
        return (self._get_obs(), reward, done, {"prob" : prob})

    def _get_obs(self):
        obs = np.zeros(self.nS)
        if self.state == LEFT_GRAY or self.state == RIGHT_GRAY:
            i = 6
        else:
            i = self.state
        obs[i] = 1
        return obs
