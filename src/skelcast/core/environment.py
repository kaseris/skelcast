import os
import logging

import randomname
import yaml

import torch
import torch.optim as optim
from torch.utils.data import random_split

from skelcast.models import MODELS
from skelcast.data import DATASETS
from skelcast.data import TRANSFORMS, COLLATE_FUNCS
from skelcast.logger import LOGGERS
from skelcast.losses import LOSSES
from skelcast.losses.torch_losses import PYTORCH_LOSSES
from skelcast.core.optimizers import PYTORCH_OPTIMIZERS

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
                 checkpoint_dir = '/home/kaseris/Documents/mount/checkpoints_forecasting',
                 train_set_size = 0.8) -> None:
        self._experiment_name = randomname.get_name()
        self.checkpoint_dir = os.path.join(checkpoint_dir, self._experiment_name)
        os.mkdir(self.checkpoint_dir)
        logging.info(f'Created checkpoint directory at {self.checkpoint_dir}')
        self.data_dir = data_dir
        self.train_set_size = train_set_size
        self.config = None
        self._model = None
        self._dataset = None
        self._train_dataset = None
        self._val_dataset = None
        self._runner = None
        self._logger = None
        self._loss = None
        self._optimizer = None
        self._collate_fn = None


    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def build_from_file(self, config_path: str) -> None:
        logging.log(logging.INFO, f'Building environment from {config_path}.')
        cfgs = read_config(config_path=config_path)
        # Build tranforms first, because they are used in the dataset
        self._transforms = build_object_from_config(cfgs.transforms_config, TRANSFORMS)
        # TODO: Add support for random splits. Maybe as external parameter?
        self._dataset = build_object_from_config(cfgs.dataset_config, DATASETS, transforms=self._transforms)
        logging.info(f'Loaded dataset from {self.data_dir}.')
        # Build the loss first, because it is used in the model
        loss_registry = LOSSES if cfgs.criterion_config.get('name') not in PYTORCH_LOSSES else PYTORCH_LOSSES
        self._loss = build_object_from_config(cfgs.criterion_config, loss_registry)
        logging.info(f'Loaded loss function {cfgs.criterion_config.get("name")}.')
        self._model = build_object_from_config(cfgs.model_config, MODELS, loss_fn=self._loss)
        logging.info(f'Loaded model {cfgs.model_config.get("name")}.')
        # Build the optimizer
        self._optimizer = build_object_from_config(cfgs.optimizer_config, PYTORCH_OPTIMIZERS, params=self._model.parameters())
        logging.info(f'Loaded optimizer {cfgs.optimizer_config.get("name")}.')
        # Build the logger
        cfgs.logger_config.get('args').update({'log_dir': os.path.join(cfgs.logger_config.get('args').get('log_dir'), self._experiment_name)})
        self._logger = build_object_from_config(cfgs.logger_config, LOGGERS)
        logging.info(f'Created runs directory at {cfgs.logger_config.get("args").get("log_dir")}')
        # Build the collate_fn
        self._collate_fn = build_object_from_config(cfgs.collate_fn_config, COLLATE_FUNCS)
        logging.info(f'Loaded collate function {cfgs.collate_fn_config.get("name")}.')
        # Split the dataset into training and validation sets
        train_size = int(self.train_set_size * len(self._dataset))
        val_size = len(self._dataset) - train_size
        self._train_dataset, self._val_dataset = random_split(self._dataset, [train_size, val_size])
        # Build the runner
        self._runner = Runner(model=self._model,
                            optimizer=self._optimizer,
                            logger=self._logger,
                            collate_fn=self._collate_fn,
                            train_set=self._train_dataset,
                            val_set=self._val_dataset,
                            checkpoint_dir=self.checkpoint_dir,
                            **cfgs.runner_config.get('args'))
        logging.info(f'Finished building environment from {config_path}.')
        self._runner.setup()
        logging.info(f'Set up runner.')

    
    def run(self) -> None:
        # Must check if there is a checkpoint directory
        # If there is, load the latest checkpoint and continue training
        # Else, create a new checkpoint directory and start training
        # If there's not a checkpoint directory, use the self._runner.fit() method
        # Otherwise, use the self._runner.resume(path_to_checkpoint) method
        return self._runner.fit()
