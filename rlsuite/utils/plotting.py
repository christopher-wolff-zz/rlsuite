"""Plotting utilities."""

from matplotlib import pyplot as plt
import seaborn as sns

from rlsuite.utils import load_data


def plot_experiment(log_dir, x, y):
    sns.set(style="whitegrid", font_scale=1.5)
    sns.lineplot(data=load_data(log_dir), x=x, y=y, ci='sd')
