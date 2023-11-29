import randomname
import yaml

import torch
from torch.utils.data import random_split

from skelcast.models import MODELS
from skelcast.data import DATASETS
from skelcast.data import TRANSFORMS
from skelcast.data import COLLATE_FUNCS
from skelcast.logger import LOGGERS

from skelcast.experiments.runner import Runner

torch.manual_seed(133742069)

class Environment:
    def __init__(self, data_dir: str = '/home/kaseris/Documents/data_ntu_rbgd',
                 checkpoint_dir = '/home/kaseris/Documents/checkpoints_forecasting') -> None:
        self._experiment_name = randomname.get_name()
        self.data_dir = data_dir
        self.config = None
        self._model = None
        self._dataset = None
        self._train_dataset = None
        self._val_dataset = None
        self._runner = None
        self._logger = None

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def build_from_file(self, config_path: str) -> None:
        config = self._parse_file(config_path)
        self.config = config
        # self._build_dataset()
        self._build_model()
        self._build_logger()

    def _build_model(self) -> None:
        model_config = self.config['model']
        _name = model_config.get('name')
        _args = model_config.get('args')
        self._model = MODELS.get_module(_name)(**_args)

    def _build_dataset(self) -> None:
        dataset_config = self.config['dataset']
        _name = dataset_config.get('name')
        _args = dataset_config.get('args')
        _transforms_cfg = dataset_config.get('args').get('transforms')
        _transforms = TRANSFORMS.get_module(_transforms_cfg.get('name'))(**_transforms_cfg.get('args'))
        _args['transforms'] = _transforms
        self._dataset = DATASETS.get_module(_name)(self.data_dir, **_args)
        # Split the dataset
        _train_len = int(self.config['train_data_percentage'] * len(self._dataset))
        self._train_dataset, self._val_dataset = random_split(self._dataset, [_train_len, len(self._dataset) - _train_len])
    
    def _build_logger(self) -> None:
        logger_config = self.config['runner']['args'].get('logger')
        logdir = os.path.join(logger_config['args']['save_dir'], self.experiment_name)
        self._logger = LOGGERS.get_module(logger_config['name'])(logdir)

    def _create_checkpoint_dir(self) -> None:
        if os.path.exists(os.path.join(self.checkpoint_dir, self._experiment_name)):
            raise ValueError(f'Checkpoint directory {self.checkpoint_dir} already exists.')
        else:
            os.mkdir(os.path.join(self.checkpoint_dir, self._experiment_name))
            # Write the config file for future reference
            with open(os.path.join(self.checkpoint_dir, self._experiment_name, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)

    def _parse_file(self, fname: str) -> None:
        with open(fname, 'r') as f:
            config = yaml.safe_load(f)
        return config

if __name__ == '__main__':
    env = Environment()
    import os
    import os.path as osp
    print(os.getcwd())
    cfg_file = osp.join(os.getcwd(), 'configs/lstm_regressor_1024x1024.yaml')
    env.build_from_file(cfg_file)
