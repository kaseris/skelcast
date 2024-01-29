from skelcast.core.registry import Registry

DATASETS = Registry()
COLLATE_FUNCS = Registry()
TRANSFORMS = Registry()

from .dataset import NTURGBDCollateFn, NTURGBDDataset
from .transforms import MinMaxScaleTransform
from .human36m import Human36MDataset