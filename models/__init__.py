from enum import Enum
import importlib
from inspect import isfunction


class MODEL(Enum):
    UNET = 'unet'
    ATTEN_UNET = 'attentionUnet'
    UNET_PP = 'unetPlusPlus'
    DEEPLABV3 = 'deeplabv3'
    DEEPLABV_U = 'deeplab_u'
    SEGNET = 'segnet'
    PSPNET = 'PSPNet.pspnet'
    FCN8s = 'FCN.fcn8s'
    FCN16s = 'FCN.fcn16s'
    FCN32s = 'FCN.fcn32s'


def get_model(config):
    name = config['name']
    model = MODEL[name].value
    module_fp = f'models.{model}'

    # 加载模块
    module = importlib.import_module(module_fp)
    assert hasattr(module, 'create_model')
    obj = getattr(module, 'create_model')
    assert isfunction(obj)
    return obj(config)

