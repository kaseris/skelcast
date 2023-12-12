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
from skelcast.core.config import read_config, build_object_from_config

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
        logging.log(logging.INFO, f'Building environment from {config_path}.')
        cfgs = read_config(config_path=config_path)
        self._transforms = build_object_from_config(cfgs.transforms_config, TRANSFORMS)
        print(self._transforms)
        # TODO: Add support for random splits
        self._dataset = build_object_from_config(cfgs.dataset_config, DATASETS)
        
        
    
    def run(self) -> None:
        # Must check if there is a checkpoint directory
        # If there is, load the latest checkpoint and continue training
        # Else, create a new checkpoint directory and start training
        # If there's not a checkpoint directory, use the self._runner.fit() method
        # Otherwise, use the self._runner.resume(path_to_checkpoint) method
        return self._runner.fit()

if __name__ == '__main__':
    format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format)
    env = Environment()
    env.build_from_file('configs/pvred.yaml')
    # env.run()