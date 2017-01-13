local display = require 'display'
local image = require 'image'

function scale(tensor, factor)
   factor = factor or 2
   local ret = nil
   if tensor:dim() == 2 then
      ret = image.scale(tensor, tensor:size(1)*factor, tensor:size(2)*factor, "simple")
   elseif tensor:dim() == 4 then
      tensor = tensor:reshape(tensor:size(3), tensor:size(4))
      ret = scale(tensor, factor)
      ret = ret:reshape(1, 1, ret:size(1), ret:size(2))
   end

   return ret
end

function scaleFilter(filter, factor)   
   local scaledFilter = torch.Tensor(filter:size(1), filter:size(2), factor*filter:size(3), 
      factor*filter:size(4)):zero()
   
   for i=1,filter:size(1) do
      for j=1,filter:size(2) do
         scaledFilter[{{i},{j},{},{}}] = scale(filter[{{i},{j},{},{}}], factor)
      end
   end

   return scaledFilter
end

local opt = {}
local omniglot = {28, 1}
local imagenet = {84, 3}
opt.homePath = 'cortex-core.projects.research.oneShotLSTM.'
opt.nClasses = 5
opt.nIn = omniglot[1]
opt.nDepth = omniglot[2]

--parse command line arguments
cmd = torch.CmdLine()
cmd:option('--params', '', 'meta-learner params')
cmd:option('--net', '', 'learner net')

local argVals = cmd:parse(arg)
local params = torch.load(argVals.params)

local model = require(opt.homePath .. argVals.net)({
      nClasses=opt.nClasses,
      classify=true,
      useCUDA=opt.useCUDA,
      nIn=opt.nIn,
      nDepth=opt.nDepth
   })

print('Meta Learner params: ')
print(params)

netParams, gParams = model.net:getParameters()
print('Learner net size: ')
print(netParams:size())
netParams:copy(params[2].cI)

-- unflatten cI
local firstLayerW = model.net.modules[1].weight 
print(firstLayerW:size())
display.image(scaleFilter(firstLayerW, 10))
local secondLayerW = model.net.modules[5].weight:index(1, torch.LongTensor({1,2,3,4,5}))
print(secondLayerW:size())
display.image(scaleFilter(secondLayerW, 10))
local thirdLayerW = model.net.modules[9].weight:index(1, torch.LongTensor({1,2,3,4}))
print(thirdLayerW:size())
display.image(scaleFilter(thirdLayerW, 10))

