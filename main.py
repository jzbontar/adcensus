import sys
import pickle

from scipy.ndimage.filters import median_filter
import numpy as np
from PIL import Image

import pyximport; pyximport.install()
import main_

DEBUG = 1

def match(x0, x1):
    x0m = median_filter(x0, size=(3, 3, 1))
    x1m = median_filter(x1, size=(3, 3, 1))

    # ad
    ad_vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        ad_vol[i,:,i:] = np.mean(np.abs(x0m[:,i:] - x1m[:,:width - i]), 2)

    if DEBUG:
        pred = ad_vol.argmin(0).astype(np.float64) * scale
        Image.fromarray(pred.astype(np.uint8)).save('report/img/ad_vol.png')

    # census
    x0c = main_.census_transform(x0m)
    x1c = main_.census_transform(x1m)
    census_vol = np.ones((disp_max, height, width)) * np.inf
    for i in range(disp_max):
        census_vol[i,:,i:] = np.sum(x0c[:,i:] != x1c[:,:width - i], 2)
    census_vol /= 3

    if DEBUG:
        pred = census_vol.argmin(0).astype(np.float64) * scale
        Image.fromarray(pred.astype(np.uint8)).save('report/img/census_vol.png')

    # adcensus
    def rho(c, lambda_):
        return 1 - np.exp(-c / lambda_)

    ad_vol_robust = rho(ad_vol, 10)
    census_vol_robust = rho(census_vol, 30)
    adcensus_vol = ad_vol_robust + census_vol_robust

    if DEBUG:
        pred = adcensus_vol.argmin(0).astype(np.float64) * scale
        Image.fromarray(pred.astype(np.uint8)).save('report/img/adcensus_vol.png')

    # cbca
    x0c = main_.cross(x0m)
    x1c = main_.cross(x1m)

    for i in range(2):
        adcensus_vol = main_.cbca(x0c, x1c, adcensus_vol, 0)
        adcensus_vol = main_.cbca(x0c, x1c, adcensus_vol, 1)
        
    if DEBUG:
        pred = adcensus_vol.argmin(0).astype(np.float64) * scale
        Image.fromarray(pred.astype(np.uint8)).save('report/img/cbca_vol.png')

    # semi-global matching
    c2_vol = main_.sgm(x0m, x1m, adcensus_vol)

    if DEBUG:
        pred = c2_vol.argmin(0).astype(np.float64) * scale
        Image.fromarray(pred.astype(np.uint8)).save('report/img/sgm_vol.png')

    return c2_vol

stereo_pairs = [['tsukuba', 16, 16], ['venus', 20, 8], ['teddy', 60, 4], ['cones', 60, 4]]
stereo_pairs = [['cones', 60, 4]]
stereo_pairs = [['tsukuba', 16, 16]]
for pair_name, disp_max, scale in stereo_pairs:
    print(pair_name)
    x0 = np.array(Image.open('data/stereo-pairs/%s/imL.png' % pair_name), dtype=np.float64)
    x1 = np.array(Image.open('data/stereo-pairs/%s/imR.png' % pair_name), dtype=np.float64)
    height = x0.shape[0]
    width = x0.shape[1]
    main_.init(height, width, disp_max)
    x0m = median_filter(x0, size=(3, 3, 1))
    x1m = median_filter(x1, size=(3, 3, 1))
    x0c = main_.cross(x0m)
    x1c = main_.cross(x1m)

    c2_1 = match(x1[:,::-1], x0[:,::-1])[:,:,::-1]
    c2_0 = match(x0, x1)

    d0 = np.argmin(c2_0, 0)
    d1 = np.argmin(c2_1, 0)

    outlier = main_.outlier_detection(d0, d1)

    if DEBUG:
        img = x0m.copy()
        img[outlier != 0] = 0
        img[outlier == 1, 0] = 255
        img[outlier == 2, 1] = 255
        Image.fromarray(img.astype(np.uint8)).save('report/img/outlier.png')

    for i in range(5):
        d0, outlier = main_.iterative_region_voting(x0c, x1c, d0, outlier)

    if DEBUG: Image.fromarray((d0 * scale).astype(np.uint8)).save('report/img/iterative_region_voting.png')
    d0 = main_.proper_interpolation(x0m, d0, outlier)
    if DEBUG: Image.fromarray((d0 * scale).astype(np.uint8)).save('report/img/proper_interpolation.png')
    d0 = main_.depth_discontinuity_adjustment(d0, c2_0)
    if DEBUG: Image.fromarray((d0 * scale).astype(np.uint8)).save('report/img/depth_discontinuity_adjustment.png')
    d0 = main_.subpixel_enhancement(d0, c2_0)
    if DEBUG: Image.fromarray((d0 * scale).astype(np.uint8)).save('report/img/subpixel_enhancement.png')
    d0 = median_filter(d0, size=3)

    pred = d0.astype(np.float64) * scale
    Image.fromarray(pred.astype(np.uint8)).save('res.py/%s.png' % pair_name)
