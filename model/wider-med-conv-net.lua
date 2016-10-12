local t = require 'torch'
local nn = require 'nn'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'

function convLayer(net, nInput, nOutput, k) 
	net:add(nn.SpatialConvolution(nInput, nOutput, k, k, 1, 1, 1, 1))
	net:add(nn.SpatialBatchNormalization(nOutput, 1e-3))
	net:add(nn.ReLU(true))
	net:add(nn.SpatialMaxPooling(2,2))
end

return function(opt)		
	local model = {}
   
	local finalSize = math.floor(opt.nIn/(2*2*2))	
	local net = nn.Sequential()	
	convLayer(net, opt.nDepth, 24, 3)
	convLayer(net, 24, 24, 3)
	convLayer(net, 24, 24, 3)
	net:add(nn.Reshape(24*finalSize*finalSize))
	
	local criterion = nil
	if opt.classify then 
		net:add(nn.Linear(24*finalSize*finalSize, opt.nClasses))
		criterion = nn.CrossEntropyCriterion()
	end
		
	model.net = util.localize(net, opt)
	model.criterion = util.localize(criterion, opt)
	model.nParams = net:getParameters():size(1) 
	model.outSize = 24*finalSize*finalSize

	print('created net:')
	print(model.net)

	return model	
end
