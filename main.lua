require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

height = 288
width = 384
disp_max = 16
ad_lambda = 10
census_lambda = 30

L1 = 34
L2 = 17
tau1 = 20
tau2 = 6

pi1 = 1
pi2 = 3
tau_so = 15

cutorch.setDevice(1)

function savePNG(fname, vol)
   local pred = torch.CudaTensor(1, 1, height, width)
   adcensus.spatial_argmin(vol, pred)
   pred:add(-1):div(disp_max)
   image.savePNG(fname, pred[{1,1}])
end

function match(x0m, x1m, x0c, x1c)
   -- ad volume
   ad_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.ad(x0m, x1m, ad_vol)

   -- census volume
   census_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.census(x0m, x1m, census_vol)

   -- adcensus volume
   function rho(c, lambda)
      c:mul(-1 / lambda):exp():mul(-1):add(1)
   end
   rho(ad_vol, ad_lambda)
   rho(census_vol, census_lambda)
   adcensus_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus_vol:add(ad_vol, 1, census_vol)

   -- cbca
   tmp = torch.CudaTensor(1, disp_max, height, width)
   adcensus.cbca(x0c, x1c, adcensus_vol, tmp)

   -- sgm
   tmp = torch.CudaTensor(8, disp_max, height, width):zero()
   adcensus.sgm(x0m, x1m, adcensus_vol, tmp, pi1, pi2, tau_so)
   sgm_out = tmp:sum(1)

   return sgm_out
end

x0 = image.loadPNG('data/tsukuba0.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()
x1 = image.loadPNG('data/tsukuba1.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()

-- median filter
x0m = torch.CudaTensor(1, 3, height, width)
x1m = torch.CudaTensor(1, 3, height, width)
adcensus.median3(x0, x0m)
adcensus.median3(x1, x1m)

-- cross computation
x0c = torch.CudaTensor(1, 4, height, width)
x1c = torch.CudaTensor(1, 4, height, width)
adcensus.cross(x0m, x0c, L1, L2, tau1, tau2)
adcensus.cross(x1m, x1c, L1, L2, tau1, tau2)

-- fliplr
x0mf = torch.CudaTensor(1, 3, height, width)
x1mf = torch.CudaTensor(1, 3, height, width)
x0cf = torch.CudaTensor(1, 4, height, width)
x1cf = torch.CudaTensor(1, 4, height, width)

adcensus.fliplr(x0m, x0mf)

-- c2_0 = match(x0m, x1m, x0c, x1c)
-- c2_1 = match(x0mf, x1mf, x0cf, x1cf)

image.savePNG('bar.png', x0mf[1]:div(255))
