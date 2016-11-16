local t = require 'torch'
local nn = require 'nn'
local util = require 'util.util'

function convLayer(net, nInput, nOutput, k, useDropout) 
   if useDropout == nil then useDropout = true end
   
   net:add(nn.SpatialConvolution(nInput, nOutput, k, k, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(nOutput, 1e-3))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(2,2))
   if useDropout then
      net:add(nn.Dropout(0.1))
   end
end

return function(opt)    
   local nFilters = 64
   local useDropout = false 
   local finalSize = math.floor(opt.nIn/(2*2*2*2))

   local model = {}
   local net = nn.Sequential()   
   convLayer(net, opt.nDepth, nFilters, 3, useDropout)
   convLayer(net, nFilters, nFilters, 3, useDropout)
   convLayer(net, nFilters, nFilters, 3, useDropout)
   convLayer(net, nFilters, nFilters, 3, useDropout)
   net:add(nn.Reshape(nFilters*finalSize*finalSize))
   
   local criterion = nil
   if opt.classify then 
      net:add(nn.Linear(nFilters*finalSize*finalSize, opt.nClasses))
      criterion = nn.CrossEntropyCriterion()
   end
      
   model.net = util.localize(net, opt)
   model.criterion = util.localize(criterion, opt)

   model.nParams = net:getParameters():size(1)
   model.outSize = nFilters*finalSize*finalSize 
 
   --[[model.unflattenParams = function(params) 
      return {
         t.view(t.narrow(params, 1, 1, 64*1*3*3), 64, 1, 3, 3),
         t.view(t.narrow(params, 1, 64*1*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*2+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*3+1, 64*64*3*3), 64, 64, 3, 3),
         t.view(t.narrow(params, 1, 64*1*3*3+64*3+64*64*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*4+64*64*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*5+64*64*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*6+64*64*3*3+1, 64*64*3*3), 64, 64, 3, 3),
         t.view(t.narrow(params, 1, 64*1*3*3+64*6+64*64*3*3*2+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*7+64*64*3*3*2+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*8+64*64*3*3*2+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*9+64*64*3*3*2+1, 64*64*3*3), 64, 64, 3, 3),
         t.view(t.narrow(params, 1, 64*1*3*3+64*9+64*64*3*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*10+64*64*3*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*11+64*64*3*3*3+1, 64), 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*12+64*64*3*3*3+1, 5*64), 5, 64),
         t.view(t.narrow(params, 1, 64*1*3*3+64*12+64*64*3*3*3+5*64+1, 5), 5)
      }
   end--]]

   print('created net:')
   print(model.net)

   return model   
end
