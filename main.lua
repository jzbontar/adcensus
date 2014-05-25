require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

height = 288
width = 384
disp_max = 16
ad_lambda = 10
census_lambda = 30

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


function combine_cost(ad_vol, ad_lambda, census_vol, census_lambda)
   -- modifies c
   function rho(c, lambda)
      c:mul(-1 / lambda):exp():mul(-1):add(1)
   end

   rho(ad_vol, ad_lambda)
   rho(census_vol, census_lambda)

   local vol = torch.CudaTensor(1, disp_max, height, width)
   return vol:add(ad_vol, 1, census_vol)
end

x0 = image.loadPNG('data/tsukuba0.png'):resize(1, 3, height, width):cuda()
x1 = image.loadPNG('data/tsukuba1.png'):resize(1, 3, height, width):cuda()
pred = torch.CudaTensor(1, 1, height, width)

ad_vol = ad(x0, x1)
census_vol = census(x0, x1)

adcensus_vol = combine_cost(ad_vol, ad_lambda, census_vol, census_lambda)

adcensus.spatial_argmin(adcensus_vol, pred)
pred:add(-1):div(disp_max)
image.savePNG('foo.adcensus.png', pred[{1,1}])
