import logging
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

from skelcast.core.environment import Environment

args = ArgumentParser()
args.add_argument('--config', type=str, default='configs/lstm_regressor_1024x1024.yaml')
args.add_argument('--data_dir', type=str, default='data')
args.add_argument('--checkpoint_dir', type=str, default='/home/kaseris/Documents/mount/checkpoints_forecasting')

args = args.parse_args()


if __name__ == '__main__':
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    
    CONTEXT_SIZE = 8

    # Maybe we won't need the Environment interface
    env = Environment(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir)
    env.build_from_file(args.config)
    dataset = env.dataset
    # hard code the checkpoint path
    checkpoint_path = '/home/kaseris/Documents/mount/checkpoints_forecast/acidic-plan/checkpoint_epoch_9_2023-12-01_192123.pt'
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    model = env.model.to('cpu')
    model.load_state_dict(model_state_dict)
    model.eval()
    sample, label = dataset[4]
    
    seq_len, n_bodies, n_joints, n_dims = sample.shape
    sample = sample.view(seq_len, n_bodies * n_joints * n_dims)
    
    sample = sample.unsqueeze(0)
    context = sample[:, :CONTEXT_SIZE, :]
    
    print(f'context shape: {context.shape}')
    # Make a forecast
    # The forecast method should be implemented in the model
    # For now let's implement it here
    # The forecast routine takes a historical record as input and returns a prediction
    # The prediction is a tensor of shape (1, CONTEXT_SIZE, n_bodies * n_joints * n_dims)
    # The CONTEXT_SIZE-th element of the prediction is the forecast fore the next time step
    # Then the prediction is appended to the historical record and the oldest element is removed
    # The process is repeated until the desired number of predictions is made
    def forecast(model, sample, n_preds):
        preds = []
        for i in range(n_preds):
            pred = model(sample.to(torch.float32))
            # print(f'pred shape: {pred.shape}')
            preds.append(pred[:, -1, :].detach().unsqueeze(1))
            sample = torch.cat([sample[:, 1:, :], pred[:, -1, :].detach().unsqueeze(1)], dim=1)
        return torch.cat(preds, dim=1)
    preds = forecast(model, context, 8)
    print(f'preds shape: {preds.shape}')
    preds = preds.view(CONTEXT_SIZE, n_bodies, n_joints, n_dims).detach()
    context = context.view(CONTEXT_SIZE, n_bodies, n_joints, n_dims)
    sample = sample.view(seq_len, n_bodies, n_joints, n_dims)
    print(f'context shape: {context.shape}')
    plt.figure(figsize=(12, 9))
    plt.plot(preds[:, 0, 0, 0])
    plt.plot(sample[CONTEXT_SIZE:CONTEXT_SIZE+8, 0, 0, 0])
    print(f'forecasted: {preds[:, 0, 0, 0]}')
    print(f'actual: {sample[CONTEXT_SIZE:CONTEXT_SIZE+8, 0, 0, 0]}')
    # print(f'abs difference between forecast and actual: {torch.abs(preds[:, 0, 0, 0] - sample[CONTEXT_SIZE:CONTEXT_SIZE+8, 0, 0, 0])}')
    # plt.legend(['Actual', 'Forecast'])
    plt.show()
