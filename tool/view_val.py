import os
import cv2 as cv
import pandas as pd

root = '/marine-farm-seg/gt'
img_fn = os.path.join(root, '1739.png')
img = cv.imread(img_fn, cv.IMREAD_UNCHANGED)
u = pd.Series(img.flatten())
print(u.unique())