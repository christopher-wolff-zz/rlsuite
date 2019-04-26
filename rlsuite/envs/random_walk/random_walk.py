import io
import sys

from gym import Env, spaces
from gym.utils import seeding
import numpy as np


# Actions
LEFT = 0
RIGHT = 1


def categorical_sample(prob_n, np_random):
    """Sample from a categorical distribution."""
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class RandomWalk(Env):

    def __init__(self, num_states=15):
        assert num_states > 0, 'num_states must be greater than 0'

        self.nS = num_states + 2  # the additional goal states
        self.nA = 2

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        leftmost_state = 0
        rightmost_state = self.nS - 1

        self.P = {
            leftmost_state: {
                LEFT: [(1.0, leftmost_state, 0.0, True)],
                RIGHT: [(1.0, leftmost_state, 0.0, True)],
            },
            rightmost_state: {
                LEFT: [(1.0, rightmost_state, 1.0, True)],
                RIGHT: [(1.0, rightmost_state, 1.0, True)],
            }
        }

        for i in range(1, self.nS - 1):
            self.P[i] = {
                LEFT: [(1.0, i - 1, 0.0, False)],
                RIGHT: [(1.0, i + 1, 0.0, False)],
            }

        # The initial state is always in the center
        self.isd = np.zeros(self.nS)
        self.isd[self.nS // 2] = 1

        self.np_random, _ = seeding.np_random(0)
        self.state = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        return None

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
        obs[self.state] = 1
        return obs

    def render(self, mode='human', close=False):
        if close:
            return

        for i in range(self.nS):
            if i == self.state:
                sys.stdout.write(' x ')
            else:
                sys.stdout.write(' _ ')
        sys.stdout.write('\n')
