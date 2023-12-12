import logging
from argparse import ArgumentParser

from skelcast.core.environment import Environment

args = ArgumentParser()
args.add_argument('--config', type=str, default='../configs/lstm_regressor_1024x1024.yaml')
args.add_argument('--data_dir', type=str, default='data')
args.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args.add_argument('--train_set_size', type=float, default=0.8, required=False)

args = args.parse_args()


if __name__ == '__main__':
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    
    env = Environment(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir)
    env.build_from_file(args.config)
    env.run()
