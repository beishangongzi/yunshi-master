from .FarmDataset import FarmDataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from util import get_map_val


def get_dataloader(config: dict, subset='train'):
    assert 'root' in config.keys()
    root = get_map_val(config, 'root', '')
    has_gt = get_map_val(config, 'has_gt', True)
    has_seg = get_map_val(config, 'has_seg', False)

    if subset not in config:
        return None

    subset_o = config[subset]
    subset_v = subset_o['meta_data'] if 'meta_data' in subset_o else None
    shuffle = subset_o['shuffle']

    batch_size = get_map_val(subset_o, 'batch_size', 2)
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    target_trans = transforms.Compose([
        lambda x: x[0, :, :]
    ])
    dataset = FarmDataset(root, subset=subset_v, has_gt=has_gt, has_seg=has_seg, transforms=trans, target_transforms=target_trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
