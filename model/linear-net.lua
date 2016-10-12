local nn = require 'nn'
local t = require 'torch'

return function(opt) 
   local model = {}
   local net = nn.Sequential()   
   net:add(nn.Reshape(28*28))

   local localize = function(x)
      if x and opt.useCUDA then
         x = x:cuda()
      elseif x and not opt.useCUDA then
         x = x:float()
      end

      return x
   end

   local criterion = nil
   if opt.classify then
      net:add(nn.Linear(28*28, opt.nClasses))
      criterion = nn.CrossEntropyCriterion()
   end

   model.net = localize(net)
   model.criterion = localize(criterion)
   model.nParams = net:getParameters():size(1)
   model.unflattenParams = function(params) 
      return {
         t.view(t.narrow(params, 1, 1, 5*28*28), 5, 784),
         t.view(t.narrow(params, 1, 5*28*28+1, 5), 5) 
      }
   end

   return model    
end
