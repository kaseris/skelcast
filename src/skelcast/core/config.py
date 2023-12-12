import re
import logging
import yaml

from collections import OrderedDict
from typing import Any, List

from skelcast.core.registry import Registry
from skelcast.data.transforms import Compose


class Config:
    def __init__(self):
        self._config = OrderedDict()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        if isinstance(value, list):
            self._config[key] = []
            for v in value:
                self._config[key].append(v)
        else:
            self._config[key] = value

    def __str__(self) -> str:
        s = self.__class__.__name__ + '(\n'
        for key, val in self._config.items():
            if isinstance(val, dict):
                s += f'\t{key}: \n'
                for k, v in val.items():
                    s += f'\t\t{k}: {v}\n'
                s += '\t\n'
            else:
                s += f'\t{key}: {val}\n'
        s += ')'
        return s
        

class ModelConfig(Config):
    def __init__(self):
        super(ModelConfig, self).__init__()


class DatasetConfig(Config):
    def __init__(self):
        super(DatasetConfig, self).__init__()


class TransformsConfig(Config):
    def __init__(self, transforms):
        super().__init__()
        self.set('args', self.parse_transforms(transforms))

    def parse_transforms(self, transforms):
        parsed_transforms = []
        for transform in transforms:
            # Assuming each transform in the list is a dictionary
            transform_dict = {'name': transform.get('name'), 'args': transform.get('args', {})}
            parsed_transforms.append(transform_dict)
        return parsed_transforms

    def get(self, key):
        if key == 'args':
            return self._config['args']
        else:
            return super().get(key)


class LoggerConfig(Config):
    def __init__(self):
        super(LoggerConfig, self).__init__()


class OptimizerConfig(Config):
    def __init__(self):
        super(OptimizerConfig, self).__init__()

    
class SchedulerConfig(Config):
    def __init__(self):
        super(SchedulerConfig, self).__init__()


class CriterionConfig(Config):
    def __init__(self):
        super(CriterionConfig, self).__init__()


class CollateFnConfig(Config):
    def __init__(self):
        super(CollateFnConfig, self).__init__()


class RunnerConfig(Config):
    def __init__(self):
        super(RunnerConfig, self).__init__()

class EnvironmentConfig:
    def __init__(self, *args) -> None:
        for arg in args:
            name = arg.__class__.__name__
            split_name = re.findall('[A-Z][^A-Z]*', name)
            name = '_'.join([s.lower() for s in split_name])
            setattr(self, name, arg)

    def __str__(self) -> str:
        s = self.__class__.__name__ + '(\n'
        for key, val in self.__dict__.items():
            s += f'\t{key}: {val}\n'
        s += ')'
        return s


def build_object_from_config(config: Config, registry: Registry, **kwargs):
    if isinstance(config, TransformsConfig):
        list_of_transforms = []
        for transform in config.get('args'):
            logging.debug(transform)
            tf = registry.get_module(transform.get('name'))(**transform.get('args'))
            list_of_transforms.append(tf)
        return Compose(list_of_transforms)
    else:
        _name = config.get('name')
        _args = config.get('args')
        _args.update(kwargs)
        return registry.get_module(_name)(**_args)

def summarize_config(configs: List[Config]):
    with open(f'/home/kaseris/Documents/mount/config.txt', 'w') as f:
        for config in configs:
            f.write(str(config))
            f.write('\n\n')


CONFIG_MAPPING = {
    'model': ModelConfig,
    'dataset': DatasetConfig,
    'transforms': TransformsConfig,
    'logger': LoggerConfig,
    'optimizer': OptimizerConfig,
    'scheduler': SchedulerConfig,
    'loss': CriterionConfig,
    'collate_fn': CollateFnConfig,
    'runner': RunnerConfig
}

def read_config(config_path: str):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    cfgs = []
    for key in data:
        if key == 'transforms':
            # Initialize TransformsConfig with the transforms data
            config = CONFIG_MAPPING[key](data[key])
            logging.debug(f'Loading {key} config. Building {config.__class__.__name__} object.')
        else:
            # Initialize other configurations normally
            config = CONFIG_MAPPING[key]()
            logging.debug(f'Loading {key} config. Building {config.__class__.__name__} object.')
            config.set('name', data[key]['name'])
            config.set('args', data[key]['args'])
        
        logging.debug(config)
        cfgs.append(config)
    return EnvironmentConfig(*cfgs)
