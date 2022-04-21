__version__ = "0.7.1"
from .model import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)
from util.MapUtil import get_map_val


def create_model(config):
    ch_in = get_map_val(config, 'ch_in', 3)
    ch_out = get_map_val(config, 'ch_out', 7)
    backbone = get_map_val(config, 'backbone', 'efficientnet-b7')
    return EfficientNet.from_name(backbone, in_channels=ch_in, num_classes=ch_out)