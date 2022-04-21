import cv2 as cv
import pandas as pd
import os
import matplotlib.pyplot as plt


root = '/marine-farm-seg2'
val_root = '../log/1/val/seg'
meta_store_fp = os.path.join(root, 'val.txt')
with open(meta_store_fp) as fp:
    files = fp.readlines()
    print(files)
    for file in files:
        img_fp = os.path.join(root, 'image', file).strip()
        gt_fp = os.path.join(root, 'gt', file).strip()
        seg_fp = os.path.join(val_root, file).strip()

        if os.path.exists(seg_fp):
            img = cv.imread(img_fp)
            gt = cv.imread(gt_fp)
            seg = cv.imread(seg_fp)
            seg[seg>0] = 255
            pics = [img, gt, seg]

            for i in range(1, len(pics)+1):
                plt.subplot(1, len(pics), i)
                plt.imshow(pics[i-1])
            plt.show()
