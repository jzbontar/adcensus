require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

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

tau_e = 10

cutorch.setDevice(1)

function savePNG(fname, vol)
   --local pred = torch.CudaTensor(1, 1, height, width)
   --adcensus.spatial_argmin(vol, pred)
   pred = vol:double():mul(scale)
   pred.libpng.save(fname, pred[{1,1}])
end

function saveOutlier(fname, x0, outlier)
   img = x0:double()
   for i=1,height do
      for j=1,width do
         if outlier[{1,1,i,j}] == 1 then
            img[{1,1,i,j}] = 0
            img[{1,2,i,j}] = 255
            img[{1,3,i,j}] = 0
         end
         if outlier[{1,1,i,j}] == 2 then
            img[{1,1,i,j}] = 255
            img[{1,2,i,j}] = 0
            img[{1,3,i,j}] = 0
         end
      end
   end
   img.libpng.save(fname, img[1])
end

function match(x0, x1)
   -- ad volume
   local ad_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.ad(x0, x1, ad_vol)

   -- census volume
   local census_vol = torch.CudaTensor(1, disp_max, height, width)
   adcensus.census(x0, x1, census_vol)

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
   adcensus.cross(x0, x0c, L1, L2, tau1, tau2)
   adcensus.cross(x1, x1c, L1, L2, tau1, tau2)

   -- cbca
   adcensus.cbca(x0c, x1c, adcensus_vol)

   -- sgm
   local tmp = torch.CudaTensor(8, disp_max, height, width):zero()
   adcensus.sgm(x0, x1, adcensus_vol, tmp, pi1, pi2, tau_so)
   local sgm_out = tmp:mean(1)

   return sgm_out, x0c, x1c
end

stereo_pairs = {{'tsukuba', 16, 16}, {'venus', 20, 8}, {'teddy', 60, 4}, {'cones', 60, 4}}
stereo_pairs = {{'venus', 20, 8}}
stereo_pairs = {{'teddy', 60, 4}}
for _, stereo_pair in ipairs(stereo_pairs) do
   pair_name = stereo_pair[1]
   disp_max = stereo_pair[2]
   scale = stereo_pair[3]

   x0 = image.loadPNG(('data/stereo-pairs/%s/imL.png'):format(pair_name), '3', 'byte')
   x1 = image.loadPNG(('data/stereo-pairs/%s/imR.png'):format(pair_name), '3', 'byte')

   height = x0:size(2)
   width = x0:size(3)

   x0 = x0:float():resize(1, 3, height, width):cuda()
   x1 = x1:float():resize(1, 3, height, width):cuda()

   c2_0, x0c, x1c = match(x0, x1)
   c2_1 = adcensus.fliplr(match(adcensus.fliplr(x1), adcensus.fliplr(x0)))

   d0 = torch.CudaTensor(1, 1, height, width)
   d1 = torch.CudaTensor(1, 1, height, width)
   adcensus.spatial_argmin(c2_0, d0)
   adcensus.spatial_argmin(c2_1, d1)
   d0:add(-1)
   d1:add(-1)

   outlier = torch.CudaTensor(1, 1, height, width):zero()
   adcensus.outlier_detection(d0, d1, outlier, disp_max)
   adcensus.iterative_region_voting(d0, x0c, x1c, outlier, tau_s, tau_h, disp_max)
   d0 = adcensus.proper_interpolation(x0, d0, outlier)
   g1, g2 = adcensus.sobel(d0)

   x = g1:double():abs():gt(10):mul(255)
   x.libpng.save('foo.png', x[{1,1}])
   os.exit()

   d0 = adcensus.depth_discontinuity_adjustment(d0, c2_0, g1, g2, tau_e)
   d0 = adcensus.subpixel_enchancement(d0, c2_0, disp_max)
   d0 = adcensus.median3(d0)

   res = d0:mul(scale):double()
   res.libpng.save(('res.lua/%s.png'):format(pair_name), res[{1,1}])
end
