import cv2 as cv
import pandas as pd
import os
import matplotlib.pyplot as plt


def read_img(fp):
    img = cv.imread(fp, cv.IMREAD_UNCHANGED)
    return img


root = '/marine-farm-seg'
val_root = '../log/1/val/seg'
meta_store_fp = os.path.join(root, 'val.txt')
with open(meta_store_fp) as fp:
    files = fp.readlines()
    for file in files:
        img_fp = os.path.join(root, 'image', file.strip())
        gt_fp = os.path.join(root, 'gt', file.strip())
        seg_fp = os.path.join(val_root, file.strip())
        img = read_img(img_fp)
        gt = read_img(gt_fp)
        seg = read_img(seg_fp)

        pics = [img, gt, seg]
        labels = [f'img-{file}', 'GT', 'Seg']
        plt.figure(figsize=(15, 5))
        for idx, p in enumerate(pics):
            plt.subplot(1, len(pics), idx+1)
            plt.title(labels[idx])
            plt.imshow(pics[idx])
        plt.show()