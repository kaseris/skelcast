import os
import logging

import randomname
import yaml

import torch
import torch.optim as optim
from torch.utils.data import random_split

from skelcast.models import MODELS
from skelcast.data import DATASETS
from skelcast.data import TRANSFORMS
from skelcast.logger import LOGGERS

from skelcast.experiments.runner import Runner

torch.manual_seed(133742069)

class Environment:
    """
    The Environment class is designed to set up and manage the environment for training machine learning models.
    It includes methods for building models, datasets, loggers, and runners based on specified configurations.
    
    Attributes:
        _experiment_name (str): A randomly generated name for the experiment.
        checkpoint_dir (str): Directory path for storing model checkpoints.
        data_dir (str): Directory path where the dataset is located.
        config (dict, optional): Configuration settings for the model, dataset, logger, and runner.
        _model (object, optional): The instantiated machine learning model.
        _dataset (object, optional): The complete dataset.
        _train_dataset (object, optional): The training subset of the dataset.
        _val_dataset (object, optional): The validation subset of the dataset.
        _runner (object, optional): The training runner.
        _logger (object, optional): The logger for recording experiment results.

    Methods:
        experiment_name: Property that returns the experiment name.
        build_from_file(config_path): Parses the configuration file and builds the dataset, model, logger, and runner.
        run(): Starts the training process, either from scratch or by resuming from the latest checkpoint.

    Usage:
        1. Initialize the Environment with data and checkpoint directories.
        2. Call `build_from_file` with the path to a configuration file.
        3. Use `run` to start the training process.

    Note:
        This class is highly dependent on external modules and configurations. Ensure that all required modules
        and configurations are properly set up before using this class.
    """
    def __init__(self, data_dir: str = '/home/kaseris/Documents/data_ntu_rbgd',
                 checkpoint_dir = '/home/kaseris/Documents/checkpoints_forecasting') -> None:
        self._experiment_name = randomname.get_name()
        self.checkpoint_dir = checkpoint_dir
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
        logging.log(logging.INFO, f'Building environment from {config_path}.')
        self._build_dataset()
        self._build_model()
        self._build_logger()
        self._build_runner()

    def _build_model(self) -> None:
        logging.log(logging.INFO, 'Building model.')
        model_config = self.config['model']
        _name = model_config.get('name')
        _args = model_config.get('args')
        self._model = MODELS.get_module(_name)(**_args)
        logging.log(logging.INFO, f'Model creation complete.')

    def _build_dataset(self) -> None:
        logging.log(logging.INFO, 'Building dataset.')
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
        logging.log(logging.INFO, f'Train set size: {len(self._train_dataset)}')
    
    def _build_logger(self) -> None:
        logging.log(logging.INFO, 'Building logger.')
        logger_config = self.config['runner']['args'].get('logger')
        logdir = os.path.join(logger_config['args']['save_dir'], self.experiment_name)
        self._logger = LOGGERS.get_module(logger_config['name'])(logdir)
        logging.log(logging.INFO, f'Logging to {logdir}.')

    def _build_runner(self) -> None:
        logging.log(logging.INFO, 'Building runner.')
        runner_config = self.config['runner']
        _args = runner_config.get('args')
        _args['logger'] = self._logger
        _args['optimizer'] = optim.AdamW(self._model.parameters(), lr=_args.get('lr'))
        _args['train_set'] = self._train_dataset
        _args['val_set'] = self._val_dataset
        _args['model'] = self._model
        _args['train_set'] = self._train_dataset
        _args['val_set'] = self._val_dataset
        _args['checkpoint_dir'] = os.path.join(self.checkpoint_dir, self._experiment_name)
        self._runner = Runner(**_args)
        self._runner.setup()
        logging.log(logging.INFO, 'Runner setup complete.')

    def _create_checkpoint_dir(self) -> None:
        if os.path.exists(os.path.join(self.checkpoint_dir, self._experiment_name)):
            raise ValueError(f'Checkpoint directory {self.checkpoint_dir} already exists.')
        else:
            logging.log(logging.INFO, f'Creating checkpoint directory: {self.checkpoint_dir}.')
            os.mkdir(os.path.join(self.checkpoint_dir, self._experiment_name))

    def _parse_file(self, fname: str) -> None:
        with open(fname, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def run(self) -> None:
        # Must check if there is a checkpoint directory
        # If there is, load the latest checkpoint and continue training
        # Else, create a new checkpoint directory and start training
        # If there's not a checkpoint directory, use the self._runner.fit() method
        # Otherwise, use the self._runner.resume(path_to_checkpoint) method
        if not os.path.exists(os.path.join(self.checkpoint_dir, self._experiment_name)):
            self._create_checkpoint_dir()
            return self._runner.fit()
        else:
            return self._runner.resume(os.path.join(self.checkpoint_dir, self._experiment_name))
