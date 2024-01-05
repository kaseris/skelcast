import argparse
import logging

import torch
import torch.nn as nn

from skelcast.data.dataset import NTURGBDDataset
from skelcast.data.transforms import MinMaxScaleTransform
from skelcast.primitives.visualize import visualize_skeleton
from skelcast.models.transformers.sttf import SpatioTemporalTransformer

argparser = argparse.ArgumentParser(description='Visualize skeleton movement.')
argparser.add_argument('--dataset', type=str, required=True, help='Path to the dataset.')
argparser.add_argument('--sample', type=int, required=True, help='Sample index to visualize.')
argparser.add_argument('--cache-file', type=str, required=False, help='Path to the cache file.')

args = argparser.parse_args()


if __name__ == '__main__':
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    tf = MinMaxScaleTransform(feature_scale=[0.0, 1.0])
    dataset = NTURGBDDataset(args.dataset, missing_files_dir='data/missing/', label_file='data/labels.txt',
                            cache_file=args.cache_file,
                            max_number_of_bodies=1, transforms=tf)
    model = SpatioTemporalTransformer(n_joints=25, d_model=256, n_blocks=3, n_heads=8, d_head=16, mlp_dim=512, loss_fn=nn.SmoothL1Loss(), dropout=0.5)
    # TODO: Remove the hard coding of the checkpoint path
    checkpoint = torch.load('/home/kaseris/Documents/mount/checkpoints_forecasting/presto-class/checkpoint_epoch_16_2024-01-05_092620.pt')
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    model = model.to('cpu')
    skeleton, label = dataset[args.sample]
    seq_len, n_bodies, n_joints, n_dims = skeleton.shape
    # input_to_model = skeleton.unsqueeze(0)
    # preds, _ = model(input_to_model.to(torch.float32), y=None, masks=None)
    preds = model.predict(skeleton.to(torch.float32), n_steps=30, observe_from_to=[1, 11])
    logging.info(f'preds shape: {preds.shape}')
    visualize_skeleton(skeleton.squeeze(1), trajectory=preds.squeeze(0), framerate=5)
    # preds = preds.view(preds.shape[1], 1, 25, 3)
    # TODO: Visualize the prediction superimposed on the skeleton
    # visualize_skeleton(preds.detach().squeeze(0))