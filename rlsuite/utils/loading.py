"""Utilities for loading and aggregating log data."""

import os

import pandas as pd


def load_data(root_dir, file_name='statistics.tsv', cols=None):
    """Load statistics from multiple runs of an experiment.

    Recursively walks through `root_dir` to find all folders with a file called
    `file_name`. Each such folder is considered a run.

    This method makes it possible to aggregate data from the same experiment
    with different seeds, but is also suitable if there is just a single run.

    Args:
        root_dir (str): A root directory containing the runs.
        file_name (str): The statistics file name to look for.
        cols (list): Optional. A subset of columns to load from each file.

    Returns:
        A pandas data frame containing the experiment data.

    """
    data = []
    num_files = 0
    for root, _, files in os.walk(root_dir):
        if file_name in files:
            path = os.path.join(root, file_name)
            data.append(pd.read_csv(path, sep='\t', usecols=cols, engine='python'))
            num_files += 1
    return pd.concat(data, keys=range(num_files), names=['run'])
