from .module import SkelcastModule as SkelcastModule
from skelcast.core.registry import Registry

MODELS = Registry()
ENCODERS = Registry()
DECODERS = Registry()

from .rnn.lstm import SimpleLSTMRegressor
from .transformers.transformer import ForecastTransformer
from .rnn.pvred import PositionalVelocityRecurrentEncoderDecoder
from .rnn.pvred import Encoder, Decoder
from .cnn.unet import Unet