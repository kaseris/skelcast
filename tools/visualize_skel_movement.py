import argparse
import logging

from skelcast.data.dataset import NTURGBDDataset
from skelcast.primitives.visualize import visualize_skeleton

argparser = argparse.ArgumentParser(description='Visualize skeleton movement.')
argparser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.')
argparser.add_argument('--sample', type=int, required=True, help='Sample index to visualize.')
argparser.add_argument('--cache-file', type=str, required=False, help='Path to the cache file.')

args = argparser.parse_args()


if __name__ == '__main__':
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    
    dataset = NTURGBDDataset(args.dataset, missing_files_dir='data/missing/', label_file='data/labels.txt',
                            cache_file=args.cache_file,
                            max_number_of_bodies=1)
    skeleton, label = dataset[args.sample]
    logging.info(f'Label: {label}')
    visualize_skeleton(skeleton.squeeze(1))
