import sys
import pickle

from scipy.ndimage.filters import median_filter
import numpy as np
from PIL import Image

import pyximport; pyximport.install()
import main_

height = 288
width = 384
disp_max = 16

def match(x0, x1):
    x0m = median_filter(x0, size=(3, 3, 1))
    x1m = median_filter(x1, size=(3, 3, 1))

    # ad
    ad_vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        ad_vol[i,:,i:] = np.mean(np.abs(x0m[:,i:] - x1m[:,:width - i]), 2)

    # census
    x0c = main_.census_transform(x0m)
    x1c = main_.census_transform(x1m)
    census_vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        census_vol[i,:,i:] = np.sum(x0c[:,i:] != x1c[:,:width - i], 2)
    census_vol /= 3

    # adcensus
    def rho(c, lambda_):
        return 1 - np.exp(-c / lambda_)

    ad_vol_robust = rho(ad_vol, 10)
    census_vol_robust = rho(census_vol, 30)
    adcensus_vol = ad_vol_robust + census_vol_robust

    # cbca
    x0c = main_.cross(x0m)
    x1c = main_.cross(x1m)

    for i in range(2):
        adcensus_vol = main_.cbca(x0c, x1c, adcensus_vol, 0)
        adcensus_vol = main_.cbca(x0c, x1c, adcensus_vol, 1)

    # semi-global matching
    c2_vol = main_.sgm(x0m, x1m, adcensus_vol)
    return c2_vol

x0 = np.array(Image.open('data/tsukuba0.png'), dtype=np.float64)
x1 = np.array(Image.open('data/tsukuba1.png'), dtype=np.float64)
sys.exit()
x0m = median_filter(x0, size=(3, 3, 1))
x1m = median_filter(x1, size=(3, 3, 1))
x0c = main_.cross(x0m)
x1c = main_.cross(x1m)

c2_0 = match(x0, x1)
c2_1 = match(x1[:,::-1], x0[:,::-1])[:,:,::-1]

d0 = np.argmin(c2_0, 0)
d1 = np.argmin(c2_1, 0)

outlier = main_.outlier_detection(d0, d1)

for i in range(4):
    d0, outlier = main_.iterative_region_voting(x0c, x1c, d0, outlier)

d0 = main_.proper_interpolation(x0m, d0, outlier)
d0 = main_.depth_discontinuity_adjustment(d0, c2_0)
d0 = main_.subpixel_enchancement(d0, c2_0)
d0 = median_filter(d0, size=3)

pred = d0.astype(np.float64) * 255 / disp_max
Image.fromarray(pred.astype(np.uint8)).save('foo.png')
sys.exit()
