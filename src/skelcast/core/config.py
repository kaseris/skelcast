import abc
import logging
import yaml

from typing import List

from skelcast.core.registry import Registry


class Config:
    def __init__(self):
        self._config = {}
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
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
    def __init__(self):
        super(TransformsConfig, self).__init__()


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


def build_object_from_config(config: Config, registry: Registry, **kwargs):
    _name = config.get('name')
    _args = config.get('args')
    _args.update(kwargs)
    return registry[_name](**_args)

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
        config = CONFIG_MAPPING[key]()
        logging.debug(f'Loading {key} config. Building {config.__class__.__name__} object.')
        config.set('name', data[key]['name'])
        config.set('args', data[key]['args'])
        logging.debug(config)
        cfgs.append(config)
    return cfgs
