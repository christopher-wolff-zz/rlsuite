# Reinforcement Learning Suite

A collection of basic reinforcement learning algorithms and environments. All algorithms are self-contained and implemented using TensorFlow 2.0+.

## Installing

As of now, the package isn't published in PyPI yet, but you can still install it via

```
git clone https://github.com/christopher-wolff/rlsuite.git
pip install rlsuite
```

This will automatically install any required dependencies.

## Usage

There are three main modules in RL Suite: `rlsuite.algos`, `rlsuite.envs`, and `rlsuite.utils`. Algorithms are simply methods that you can import from `rlsuite.algos`. Environments can be used by importing the respective module and then calling `gym.make(env_name)`. The utilities module provides tools for logging, loading experiment data, and plotting results.

The following snippet demonstrates an example workflow, in which we run the REINFORCE algorithm on the cart-pole environment described by Barto, Sutton, and Anderson (2018).

```
import gym
import tensorflow as tf
from rlsuite.algos import reinforce

reinforce(
    env_fn=lambda: gym.make('CartPole-v1'),
    policy=tf.keras.Sequential([
        layers.Dense(2, input_shape=(4,), activation='softmax'),
    ]),
    num_episodes=100,
    data_dir='/tmp/experiments/reinforce',
    log_freq=10,
)
```

## Examples

The following is a list of Jupyter notebooks that run through usage examples of RL Suite.

- [Visualization](examples/visualization.ipynb)

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
