import pyximport; pyximport.install()
import main_
import numpy as np
from PIL import Image

height = 288
width = 384
disp_max = 16

def ad(x0, x1):
    vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        vol[i,:,i:] = np.mean(np.abs(x0[:,i:] - x1[:,:width - i]), 2)
    return vol

def census_transform(x0, x1):
    x0c = main_.census_transform(x0)
    x1c = main_.census_transform(x1)
    vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        vol[i,:,i:] = np.sum(x0c[:,i:] != x1c[:,:width - i], 2)
    return vol

x0 = np.array(Image.open('data/tsukuba0.png'), dtype=np.float64)
x1 = np.array(Image.open('data/tsukuba1.png'), dtype=np.float64)

# ad_vol = ad(x0, x1)
# census_vol = main_.census(x0, x1)
census_vol = census_transform(x0, x1)

pred = np.argmin(census_vol, 0).astype(np.float64) * 255 / disp_max
Image.fromarray(pred.astype(np.uint8)).save('foo.py.png')
