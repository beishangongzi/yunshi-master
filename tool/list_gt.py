import os
import cv2 as cv
import pandas as pd

root = '/marine-farm-seg/'
# out_dir = '/marine-farm-seg2'
out_dir = '/marine-farm-seg3'

# 1. 把图片分类
multi_label = []
single_label = []

files = os.listdir(os.path.join(root, 'gt'))
for file in files:
    fp = os.path.join(root, 'gt', file)
    img = cv.imread(fp, cv.IMREAD_UNCHANGED)
    s = pd.Series(img.flatten())
    un = s.unique()
    if len(un) > 1:
        multi_label.append(file)
    else:
        single_label.append(file)

print(f'multi: {len(multi_label)}  single: {len(single_label)}')
print(multi_label)
print(single_label)


def check_dir(fp):
    if not os.path.exists(fp):
        os.mkdir(fp)
        print(f'mkdir: {fp}')


# 2. 数据拷贝到新目录 & 数据增强
img_dir = os.path.join(out_dir, 'image')
gt_dir = os.path.join(out_dir, 'gt')
check_dir(out_dir)
check_dir(img_dir)
check_dir(gt_dir)


def save_raw(file):
    img_fp = os.path.join(root, 'image', file)
    gt_fp = os.path.join(root, 'gt', file)

    img = cv.imread(img_fp, cv.IMREAD_UNCHANGED)
    gt = cv.imread(gt_fp, cv.IMREAD_UNCHANGED)

    out_img_fp = os.path.join(out_dir, 'image', file)
    out_gt_fp = os.path.join(out_dir, 'gt', file)
    cv.imwrite(out_img_fp, img)
    cv.imwrite(out_gt_fp, gt)


def save_rotate(file, rotate_idx=0):
    rotates = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]

    filename = file.split('.')[0]

    img_fp = os.path.join(root, 'image', file)
    gt_fp = os.path.join(root, 'gt', file)

    img = cv.imread(img_fp, cv.IMREAD_UNCHANGED)
    gt = cv.imread(gt_fp, cv.IMREAD_UNCHANGED)
    img = cv.rotate(img, rotates[rotate_idx])
    gt = cv.rotate(gt, rotates[rotate_idx])

    out_img_fp = os.path.join(out_dir, 'image', f'{filename}_r_{rotate_idx}.png')
    out_gt_fp = os.path.join(out_dir, 'gt', f'{filename}_r_{rotate_idx}.png')
    cv.imwrite(out_img_fp, img)
    cv.imwrite(out_gt_fp, gt)


def save_flip(file, flip_code):
    filename = file.split('.')[0]

    img_fp = os.path.join(root, 'image', file)
    gt_fp = os.path.join(root, 'gt', file)

    img = cv.imread(img_fp, cv.IMREAD_UNCHANGED)
    gt = cv.imread(gt_fp, cv.IMREAD_UNCHANGED)
    img = cv.flip(img, flip_code)
    gt = cv.flip(gt, flip_code)

    out_img_fp = os.path.join(out_dir, 'image', f'{filename}_f_{flip_code}.png')
    out_gt_fp = os.path.join(out_dir, 'gt', f'{filename}_f_{flip_code}.png')
    cv.imwrite(out_img_fp, img)
    cv.imwrite(out_gt_fp, gt)


# for file in multi_label:
#     save_raw(file)
#     save_rotate(file, 0)
#     save_rotate(file, 1)
#     save_rotate(file, 2)
#     save_flip(file, 1)


for file in single_label:
    save_raw(file)
    # save_rotate(file, 0)
    # save_rotate(file, 1)
    # save_rotate(file, 2)
    # save_flip(file, 1)
