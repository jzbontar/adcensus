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

x0 = np.array(Image.open('data/tsukuba0.png'), dtype=np.float64)
x1 = np.array(Image.open('data/tsukuba1.png'), dtype=np.float64)

x0m = median_filter(x0, size=(3, 3, 1))
x1m = median_filter(x1, size=(3, 3, 1))

adcensus_vol = pickle.load(open('foo.bin'))

v1 = main_.sgm(adcensus_vol)
#v2 = main_.sgm(adcensus_vol[:,:,::-1])[:,:,::-1]
#adcensus_volT = adcensus_vol.transpose((0,2,1))
#v3 = main_.sgm(adcensus_volT).transpose((0,2,1))
#v4 = main_.sgm(adcensus_volT[:,:,::-1])[:,:,::-1].transpose((0,2,1))
#v = (v1 + v2 + v3 + v4) / 4

pred = np.argmin(v1, 0).astype(np.float64) * 255 / disp_max
Image.fromarray(pred.astype(np.uint8)).save('foo.png')
sys.exit()

# ad
ad_vol = np.ones((disp_max, height, width)) * np.inf
for i in range(disp_max):
    ad_vol[i,:,i:] = np.mean(np.abs(x0[:,i:] - x1[:,:width - i]), 2)

# census
x0c = main_.census_transform(x0)
x1c = main_.census_transform(x1)
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

pred = np.argmin(adcensus_vol, 0).astype(np.float64) * 255 / disp_max
Image.fromarray(pred.astype(np.uint8)).save('foo.png')

pickle.dump(adcensus_vol, open('foo.bin', 'w'), -1)
