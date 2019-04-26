"""A launcher script for running algorithms."""

if __name__ == '__main__':
    import argparse
    import gym

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--alg', type=str, required=True)
    known_args, unknown_args = parser.parse_known_args()
