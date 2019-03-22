import atexit
import csv
import os
import time


def _print_table(stats):
    """Nicely print key-value pairs as a table to stdout.

    Keys will be printed in the left and values in the right column.

    Args:
        stats (dict): The key-value pairs to be printed.

    """
    max_key_len = max([len(key) for key in stats])
    width_right = 15
    width_left = max(width_right, max_key_len)
    divider = '+-' + '-' * width_left + '-+-' + '-' * width_right + '-+'

    def get_format_char(value):
        if isinstance(value, int):
            return 'd'
        elif isinstance(value, float):
            return '.4f'
        else:
            return 's'

    print(divider)
    for name, value in stats.items():
        left_format = f':>{width_left}s'
        right_format = f':<{width_right}{get_format_char(value)}'
        line_format = f'| {{{left_format}}} | {{{right_format}}} |'
        line = line_format.format(name, value)
        print(line)
    print(divider)


class Logger:
    """A multi-purpose logger for experiments.

    Adapted from https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py.

    The three main use cases are
    - displaying statistics in the console while an algorithm is running,
    - saving arbitrary experiment statistics to a file, and
    - storing the experiment state to be restored later.

    """

    def __init__(
        self,
        output_dir=None,
        output_fname='statistics.tsv',
        exp_name=None,
        log_freq=1,
    ):
        """Initialize a new Logger.

        Args:
            output_dir (str): The output directory. If `None`, defaults to a
                directory of the form `/tmp/experiments/some-random-number`.
            output_fname (str): The output file name.
            exp_name (str): The name of the experiment.
            log_freq (int): The log frequency.

        """
        # Set up output file
        self.output_dir = output_dir or '/tmp/experiments/%i' % int(time.time())
        if os.path.exists(self.output_dir):
            print(f'Warning: Directory {self.output_dir} already exists! Storing info there anyway.')
        else:
            os.makedirs(self.output_dir)
        output_filepath = os.path.join(output_dir, output_fname)
        self.output_file = open(output_filepath, 'w')
        self.file_writer = csv.writer(self.output_file, delimiter='\t')
        atexit.register(self.output_file.close)

        self.exp_name = exp_name
        self.log_freq = log_freq
        self.first_row = True
        self.log_headers = None
        self.counter = 0  # keeps track of how often log_stats is called

    def log_params(self, **params):
        """Log any number of experiment parameters and display them in stdout.

        Args:
            params: A collection of {param_name: param_value} pairs.

        """
        for name, value in params.items():
            print(f'PARAM {name} {value}')

    def log_stats(self, **stats):
        """Log any number of experiment statistics.

        This will write the statistics from the most recent step to the output
        directory if one is specified.

        The statistics are also displayed in stdout at the step interval
        specified by `self.log_freq`.

        The statistics needn't change across multiple calls to this function
        as they will eventually be stored in tabular form. They also need to
        maintain the same order.

        Args:
            stats: A collection of {param_name: param_value} pairs.

        """
        if self.first_row:
            self.log_headers = list(stats.keys())
        for key in stats:
            assert key in self.log_headers, f"Can't introduce a new key that you didn't include before: {key}"

        # Write to output file
        if self.first_row:
            self.file_writer.writerow(self.log_headers)
        self.file_writer.writerow(stats.values())
        self.output_file.flush()

        # Display in stdout
        if self.counter % self.log_freq == 0:
            _print_table(stats)

        self.first_row = False
        self.counter += 1
