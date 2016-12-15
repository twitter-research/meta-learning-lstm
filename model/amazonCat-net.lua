local t = require 'torch'
local nn = require 'nn'
local util = require 'util.util'

return function(opt)    
   
   local model = {}
   local net = nn.Sequential()   
   
   local nHidden = 50
   net:add(nn.Linear(opt.nIn, nHidden)) 
   net:add(nn.ReLU(true))
   
   --net:add(nn.BatchNormalization(nHidden))
   --net:add(nn.Dropout(0.4))   
   
   --net:add(nn.Linear(nHidden, nHidden))
   --net:add(nn.ReLU(true))
   --net:add(nn.Dropout(0.4))  
 
   --net:add(nn.ReLU(true))  
   --net:add(nn.BatchNormalization(nHidden))  
   local criterion = nil
   if opt.classify then 
      net:add(nn.Linear(nHidden, opt.nClasses))
      criterion = nn.CrossEntropyCriterion()
   end
      
   model.net = util.localize(net, opt)
   model.criterion = util.localize(criterion, opt)

   model.nParams = net:getParameters():size(1)
   model.outSize = nHidden 
 
   print('created net:')
   print(model.net)

   return model   
end
