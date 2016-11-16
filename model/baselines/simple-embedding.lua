local t = require 'torch'
local autograd = require 'autograd'
local util = require 'util.util'

return function(opt)
   local model = {}

   -- load and functionalize embedding net
   local model1 = require(opt.model)({
      nClasses=opt.nClasses, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
   })
   local embedNet1 = model1.net
   local model2 = require(opt.model)({
      nClasses=opt.nClasses, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
   })
   local embedNet2 = model2.net

   embedNet1:double()
   embedNet2:double()
   local modelF, paramsF = autograd.functionalize(embedNet1) 
   local modelG, paramsG = autograd.functionalize(embedNet2) 
   model.params = {f=paramsG, g=paramsG}
   --model.params = paramsF

   -- set training or evaluate mode
   model.set = function(mode)
      if mode == 'training' then
         modelF.module:training()
         modelG.module:training()
      elseif mode == 'evaluate' then
         modelF.module:evaluate()
         modelG.module:evaluate()
      else
         error(string.format("model.set: undefined mode - %s", mode))
      end
   end 

   model.df = function(dfDefault)
      return dfDefault
   end

   model.embedX = function(params, input, gS)
      return modelF(params.f, input)
   end
   model.embedS = function(params, input)
      return modelG(params.g, input)
   end 

   model.save = function()
      local models = {modelF=embedNet1:clearState(), modelG=embedNet2:clearState()}
      torch.save('matching-net-models.th', models)
   end

   model.load = function(networkFile, opt)
      local data = torch.load(networkFile)
      modelF, _ = autograd.functionalize(util.localize(data.modelF, opt))
      modelG, _ = autograd.functionalize(util.localize(data.modelG, opt))
   end

   return model   
end


