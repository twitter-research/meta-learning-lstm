local t = require 'torch'
local nn = require 'nn'
local util = require 'util.util'

function convLayer(net, nInput, nOutput, k) 
   net:add(nn.SpatialConvolution(nInput, nOutput, k, k, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(nOutput, 1e-3))
   net:add(nn.ReLU(true))
   net:add(nn.SpatialMaxPooling(2,2))
end

return function(opt)    
   local model = {}
   local net = nn.Sequential()   
   convLayer(net, 1, 8, 3)
   convLayer(net, 8, 8, 3)
   net:add(nn.Reshape(8*7*7))
   
   local criterion = nil
   if opt.classify then 
      net:add(nn.Linear(8*7*7, opt.nClasses))
      criterion = nn.CrossEntropyCriterion()
   end
      
   model.net = util.localize(net, opt)
   model.criterion = util.localize(criterion, opt)
   model.nParams = net:getParameters():size(1)
   
   model.unflattenParams = function(params) 
      return {
         t.view(t.narrow(params, 1, 1, 8*1*3*3), 8, 1, 3, 3),
         t.view(t.narrow(params, 1, 8*1*3*3+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*2+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*3+1, 8*8*3*3), 8, 8, 3, 3),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*3+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*4+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*5+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*6+1, 5*392), 5, 392),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*6+5*392+1, 5), 5)
      }   
   end 

   --[[model.unflattenParams = function(params)
      return {
         t.view(t.narrow(params, 1, 1, 8*1*3*3), 8, 1, 3, 3),
         t.view(t.narrow(params, 1, 8*1*3*3+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8+1, 8*8*3*3), 8, 8, 3, 3),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8+1, 8), 8),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*2+1, 5*392), 5, 392),
         t.view(t.narrow(params, 1, 8*1*3*3+8*8*3*3+8*2+5*392+1, 5), 5)
      }
   end--]]

   print('created net:')
   print(model.net)

   return model   
end
