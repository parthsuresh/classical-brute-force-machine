import os

import argparse

from config_parse import parse_config
from data.data_preprocessing import process_data

def parse_args():
    parser = argparse.ArgumentParser(description='Classical Brute-Force Machine')
    parser.add_argument('--config_path', action='store',
                        required=True,
                        dest='config_path',
                        help='Enter path to TOML file')
    parser.add_argument('--data_path', action='store',
                        required=True,
                        dest='data_path',
                        help='Enter path to dataset')
    parser.add_argument('--output_path', action='store',
                        dest='output_path',
                        help='Enter path to output')
    results = parser.parse_args()
    return results

if __name__ == "__main__":
    args = parse_args()
    args.output_path = args.output_path if args.output_path else "."
    config = parse_config(args.config_path)
    data = process_data(args.data_path, config, args)
