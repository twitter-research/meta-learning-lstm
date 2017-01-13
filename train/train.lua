local nn = require 'nn'
local optim = require 'optim'
local util = require 'util.util'
_ = require 'moses'

return function(opt)  
   -- load config info for task, data, and model 
   opt = require(opt.task)(opt)
   opt = require(opt.data)(opt)
   opt = require(opt.model)(opt)
   if opt.test ~= '-' then
      opt = require(opt.test)(opt)
   end

   -- options 
   print('Training with options:', _.sort(opt))

   -- load cunn
   if opt.useCUDA then 
      require 'cunn' 
      require 'cutorch'
   end

   -- set up meta-train, meta-validation & meta-test datasets  
   print('Using data-loader: ' .. opt.dataLoader)
   local train, validation, test = require(opt.dataLoader)(opt) 
   local data = {train=train, validation=validation, test=test}   
   
   -- string->boolean 
   for k,v in pairs(opt) do
      if v=="true" or v=="false" then
         opt[k] = util.boolize(opt[k])
      end
   end
      
   -- run
   local acc = require(opt.metaLearner)(opt, data)
   return acc

end
