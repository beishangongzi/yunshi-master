import os
import numpy as np
import cv2 as cv
from PIL import Image
import torch
from torch.utils.data import Dataset
from util import FileExtFilter


class FarmDataset(Dataset):
    def __init__(self, root_dir, subset='train.txt', has_gt=True, has_seg=False, transforms=None, target_transforms=None):
        self.root_dir = root_dir
        self.subset = subset
        self.has_gt = has_gt
        self.has_seg = has_seg
        # load metastore
        self.metastore = self.load_metastore()
        self.transforms = transforms
        self.target_transforms = target_transforms

    def load_metastore(self):
        if self.subset is not None:
            meta_store_fp = os.path.join(self.root_dir, self.subset)
            with open(meta_store_fp) as fp:
                files = fp.readlines()
                return files
        else:
            return os.listdir(self.root_dir)

    def __len__(self):
        return len(self.metastore)

    def __getitem__(self, idx):
        res = {}
        # name
        img_name = self.metastore[idx].strip()
        res['name'] = img_name

        # read pic
        img_path = os.path.join(self.root_dir, 'image', img_name) if self.has_gt else os.path.join(self.root_dir, img_name)
        img = self._read_img(img_path)
        r = [str(x) for x in img.size]
        res['size'] = ','.join(r)
        if self.transforms:
            img = self.transforms(img)
        res['img'] = img

        # read gt
        if self.has_gt:
            gt_path = os.path.join(self.root_dir, 'gt', img_name)
            gt = self._read_gt(gt_path)
            if self.transforms:
                gt = self.transforms(gt)
            gt[gt>0] = 1
            if self.target_transforms:
                gt = self.target_transforms(gt)
            res['gt'] = gt

        # read seg
        if self.has_seg:
            seg_path = os.path.join(self.root_dir, 'seg', img_name)
            seg = self._read_gt(seg_path)
            if self.transforms:
                seg = self.transforms(seg)
            seg[seg > 0] = 1
            if self.target_transforms:
                seg = self.target_transforms(seg)
            res['seg'] = seg

        return res

    def _read_img(self, img_path):
        img = Image.open(img_path)
        return img

    def _read_gt(self, img_path):
        gt = Image.open(img_path)
        return gt


if __name__ == '__main__':
    root_dir = '../data/marine-farm-seg'
    dataset = FarmDataset(root_dir, subset='train.txt')
    row = dataset.__getitem__(0)
    print(row.keys())
    print(row['name'])
    print(row['img'].shape)
    print(row['gt'].shape)
    print(len(dataset))
