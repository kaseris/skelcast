import argparse
import logging

import torch
import torch.nn as nn

from skelcast.data.dataset import NTURGBDDataset
from skelcast.primitives.visualize import visualize_skeleton
from skelcast.models.rnn.pvred import PositionalVelocityRecurrentEncoderDecoder

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
    model = PositionalVelocityRecurrentEncoderDecoder(input_dim=75,
                                                      enc_hidden_dim=64,
                                                      dec_hidden_dim=64,
                                                      enc_type='lstm',
                                                      dec_type='lstm',
                                                      include_velocity=False,
                                                      pos_enc=None,
                                                      batch_first=True,
                                                      use_padded_len_mask=False,
                                                      observe_until=20,
                                                      use_std_mask=False,
                                                      loss_fn=nn.MSELoss())
    # TODO: Remove the hard coding of the checkpoint path
    checkpoint = torch.load('/home/kaseris/Documents/mount/checkpoints_forecasting/heather-head/checkpoint_epoch_99_2023-12-13_115017.pt')
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    model = model.to('cpu')
    skeleton, label = dataset[args.sample]
    seq_len, n_bodies, n_joints, n_dims = skeleton.shape
    input_to_model = skeleton.unsqueeze(0)
    preds, _ = model(input_to_model.to(torch.float32), y=None, masks=None)
    logging.info(f'preds shape: {preds.shape}')
    visualize_skeleton(skeleton.squeeze(1))
    preds = preds.view(preds.shape[1], 1, 25, 3)
    # TODO: Visualize the prediction superimposed on the skeleton
    visualize_skeleton(preds.detach().squeeze(1), framerate=5)