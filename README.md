# Reinforcement Learning Suite

A collection of basic reinforcement learning algorithms and environments. All algorithms are self-contained and implemented using TensorFlow 2.0+. The algorithms are based on pseudo-code from Sutton & Barto's book "Reinforcement learning: An introduction" (2018).

## Overview

There are three main modules in RL Suite: `rlsuite.algos`, `rlsuite.envs`, and `rlsuite.utils`. Algorithms are simply methods whose documentation can be found in the respective algorithm file. Environments can be used by importing the respective module and then calling `gym.make(env_name)`. The utilities module provides tools for logging, loading experiment data, and plotting results.

## Installing

As of now, the package isn't published in PyPI yet, but you can still install it via

```
git clone https://github.com/christopher-wolff/rlsuite.git
cd rlsuite
pip install .
```

This will automatically install any required dependencies.

## Usage

The following snippet demonstrates an example workflow, in which we run the REINFORCE algorithm (Williams, 1992) on the cart-pole environment from OpenAI Gym.

```python
import gym
import tensorflow as tf

from rlsuite.algos import reinforce
from rlsuite.utils import plot_experiment

DATA_DIR = '/tmp/experiments/reinforce'

reinforce(
    env_fn=lambda: gym.make('CartPole-v1'),
    policy=tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(4,), activation='softmax'),
    ]),
    num_episodes=100,
    data_dir=DATA_DIR,
    log_freq=10,
)
plot_experiment(DATA_DIR, x='iteration', y='episode_return')
```

## Examples

The following is a list of Jupyter notebooks that run through usage examples of RL Suite.

- [Visualization](examples/visualization.ipynb)
- [REINFORCE](examples/reinforce.ipynb)
- [Sarsa vs. Q-Learning](examples/sarsa_vs_qlearning.ipynb)
- [TD vs. MC](examples/td_vs_mc.ipynb)

## Algorithms

### Exact solution methods

- [Q-learning](rlsuite/algos/qlearning.py)
- [Sarsa](rlsuite/algos/sarsa.py)
- [Expected Sarsa](rlsuite/algos/expected_sarsa.py)
- [On-policy Monte Carlo control](rlsuite/algos/mc_control.py)
- [Policy evaluation](rlsuite/algos/policy_eval.py)

### Approximate solution methods

- [Gradient Monte Carlo Prediction](rlsuite/algos/gradient_mc_prediction.py)
- [Semi-gradient TD Prediction](rlsuite/algos/semi_gradient_td_prediction.py)
- [Least Squares Temporal Difference Learning](rlsuite/algos/lstd.py)
- [TD(lambda)](rlsuite/algos/td_lambda.py)
- [REINFORCE](rlsuite/algos/reinforce.py)

## Environments

- [Aliased Gridworld](rlsuite/envs/aliased_gridworld)
- [Gridworld](rlsuite/envs/gridworld) (from Denny Britz at https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py)
- [Random Walk](rlsuite/envs/random_walk)
