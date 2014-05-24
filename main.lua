require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

cutorch.setDevice(2)

function ad(x0, x1)
   res = torch.CudaTensor(16, height, width)
   adcensus.ad(x0, x1, res)
end

x0 = image.loadPNG('data/tsukuba0.png'):cuda()
x1 = image.loadPNG('data/tsukuba1.png'):cuda()

height = x0:size(2)
width = x0:size(3)

res_ad = ad(x0, x1)
