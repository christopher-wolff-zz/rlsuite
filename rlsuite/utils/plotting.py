"""Plotting utilities."""

from IPython.display import display
from JSAnimation.IPython_display import display_animation
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns

from rlsuite.utils import load_data


def display_frames_as_gif(frames):
    """Display a list of frames as a gif, with controls.

    Only for Jupyter notebooks.

    """
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))


def plot_experiment(log_dir, x, y):
    sns.set(style="whitegrid", font_scale=1.5)
    sns.lineplot(data=load_data(log_dir), x=x, y=y, ci='sd')
