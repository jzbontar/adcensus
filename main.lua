require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

height = 288
width = 384
disp_max = 16

cutorch.setDevice(1)

function ad(x0, x1)
   local vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.ad(x0, x1, vol)
   return vol
end

function census(x0, x1)
   local vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.census(x0, x1, vol)
   return vol
end

x0 = image.loadPNG('data/tsukuba0.png'):resize(1, 3, height, width):cuda()
x1 = image.loadPNG('data/tsukuba1.png'):resize(1, 3, height, width):cuda()
pred = torch.CudaTensor(1, 1, height, width)

ad_vol = ad(x0, x1)
census_vol = census(x0, x1)
adcensus.spatial_argmin(census_vol, pred)
pred:div(disp_max)

image.savePNG('foo.png', pred[{1,1}])
