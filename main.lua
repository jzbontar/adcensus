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

tau_s = 20
tau_h = 0.4

cutorch.setDevice(1)

function savePNG(fname, pred)
   --local pred = torch.CudaTensor(1, 1, height, width)
   --adcensus.spatial_argmin(vol, pred)
   pred = pred:clone()
   pred:div(disp_max)
   image.savePNG(fname, pred[{1,1}])
end

function match(x0m, x1m)
   -- ad volume
   local ad_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.ad(x0m, x1m, ad_vol)

   -- census volume
   local census_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.census(x0m, x1m, census_vol)

   -- adcensus volume
   function rho(c, lambda)
      c:mul(-1 / lambda):exp():mul(-1):add(1)
   end
   rho(ad_vol, ad_lambda)
   rho(census_vol, census_lambda)
   local adcensus_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus_vol:add(ad_vol, 1, census_vol)

   -- cross computation
   local x0c = torch.CudaTensor(1, 4, height, width)
   local x1c = torch.CudaTensor(1, 4, height, width)
   adcensus.cross(x0m, x0c, L1, L2, tau1, tau2)

   -- cbca
   adcensus.cbca(x0c, x1c, adcensus_vol)

   -- sgm
   local tmp = torch.CudaTensor(8, disp_max, height, width):zero()
   adcensus.sgm(x0m, x1m, adcensus_vol, tmp, pi1, pi2, tau_so)
   local sgm_out = tmp:sum(1)

   return sgm_out, x0c, x1c
end

x0 = image.loadPNG('data/tsukuba0.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()
x1 = image.loadPNG('data/tsukuba1.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()

-- median filter
x0m = adcensus.median3(x0)
x1m = adcensus.median3(x1)

c2_0, x0c, x1c = match(x0m, x1m)
c2_1 = adcensus.fliplr(match(adcensus.fliplr(x1m), adcensus.fliplr(x0m)))

d0 = torch.CudaTensor(1, 1, height, width)
d1 = torch.CudaTensor(1, 1, height, width)
adcensus.spatial_argmin(c2_0, d0)
adcensus.spatial_argmin(c2_1, d1)
d0:add(-1)
d1:add(-1)

outlier = torch.CudaTensor(1, 1, height, width)
adcensus.outlier_detection(d0, d1, outlier)
adcensus.iterative_region_voting(d0, x0c, x1c, outlier, tau_s, tau_h)
d0 = adcensus.proper_interpolation(x0m, d0, outlier)

savePNG('bar.png', d0)
os.exit()

