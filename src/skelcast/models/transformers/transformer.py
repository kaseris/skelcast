import torch
import torch.nn as nn

import skelcast.models.transformers.base as base
from skelcast.models import SkelcastModule
from skelcast.models import MODELS


@MODELS.register_module()
class ForecastTransformer(SkelcastModule):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)