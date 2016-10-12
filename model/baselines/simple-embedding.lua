local t = require 'torch'
local autograd = require 'autograd'

return function(opt)
	local model = {}

	-- load and functionalize embedding net
	local model1 = require(opt.homePath .. opt.model)({
		nClasses=opt.nClasses, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
	})
	local embedNet1 = model1.net
	local model2 = require(opt.homePath .. opt.model)({
		nClasses=opt.nClasses, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
	})
	local embedNet2 = model2.net
	local modelF, paramsF = autograd.functionalize(embedNet1)
	local modelG, paramsG = autograd.functionalize(embedNet2) 

	model.params = paramsF
	
	-- set training or evaluate mode
   model.set = function(mode)
      if mode == 'training' then
         modelF.module:training()
         modelG.module:training()
      elseif mode == 'evaluate' then
         modelF.module:evaluate()
         modelG.module:evaluate()
      end   
   end 

	model.df = function(dfDefault)
		return dfDefault
	end

	model.embedX = function(params, input, gS)
		return modelF(params, input)
	end
	model.embedS = function(params, input)
		return modelG(params, input)
	end 

	return model 	
end


