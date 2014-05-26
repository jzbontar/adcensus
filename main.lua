require 'cutorch'
require 'cunn'
require 'image'
require 'libadcensus'

height = 288
width = 384
disp_max = 18
ad_lambda = 10
census_lambda = 30
L1 = 34
L2 = 17
tau1 = 20
tau2 = 6

cutorch.setDevice(1)

x0 = image.loadPNG('data/tsukuba0.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()
x1 = image.loadPNG('data/tsukuba1.png', 3, 'byte'):float():resize(1, 3, height, width):cuda()
pred = torch.CudaTensor(1, 1, height, width)

-- ad volume
ad_vol = torch.CudaTensor(1, disp_max, height, width)
adcensus.ad(x0, x1, ad_vol)

-- census volume
census_vol = torch.CudaTensor(1, disp_max, height, width)
adcensus.census(x0, x1, census_vol)

-- adcensus volume
function rho(c, lambda)
   c:mul(-1 / lambda):exp():mul(-1):add(1)
end
rho(ad_vol, ad_lambda)
rho(census_vol, census_lambda)
adcensus_vol = torch.CudaTensor(1, disp_max, height, width)
adcensus_vol:add(ad_vol, 1, census_vol)

-- cbca
x0_t = torch.CudaTensor(1, 3, width, height):copy(x0:transpose(3, 4))
x1_t = torch.CudaTensor(1, 3, width, height):copy(x1:transpose(3, 4))
input = adcensus_vol
input_t = torch.CudaTensor(1, disp_max, width, height)
output_t = torch.CudaTensor(1, disp_max, width, height)
output = torch.CudaTensor(1, disp_max, height, width)

for i = 1,4 do
   adcensus.cbca(x0, x1, input, output, L1, L2, tau1, tau2)
   input:copy(output)
end

-- plot
adcensus.spatial_argmin(adcensus_vol, pred)
pred:add(-1):div(disp_max)
image.savePNG('foo.png', pred[{1,1}])
