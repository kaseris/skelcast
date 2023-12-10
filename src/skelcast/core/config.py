import abc

from skelcast.core.registry import Registry


class Config(metaclass=abc.ABCMeta):
    def __init__(self):
        self._config = {}

    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def set(self, key, value):
        pass


class ModelConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class DatasetConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class TransformsConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class LoggerConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class OptimizerConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value

    
class SchedulerConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class CriterionConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class CollateFnConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


class RunnerConfig(Config):
    def __init__(self):
        super().__init__()
        self._config['name'] = None
        self._config['args'] = {}

    def get(self, key):
        return self._config[key]

    def set(self, key, value):
        self._config[key] = value


def build_object_from_config(config: Config, registry: Registry, **kwargs):
    _name = config.get('name')
    _args = config.get('args')
    _args.update(kwargs)
    return registry[_name](**_args)

def summarize_config(config: Config):
    _name = config.get('name')
    _args = config.get('args')
    return f'{_name}({_args})'
