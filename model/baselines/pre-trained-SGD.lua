local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
local _ = require 'moses'

function eval(network, criterion, evalSet, conf, initParams, opt, optimOpt, updates)
	local params, gParams = network:getParameters()
	
	-- evaluate validation set
	for v=1,1 do
		local trainSet, testSet = evalSet.createEpisode({})
	
		-- get all train examples
		local trainData = trainSet:get()
		local testData = testSet:get()

		-- k-shot test loop
		_.each(conf, function(k, cM)
			local optimOptCopy = util.deepClone(optimOpt)
	
			-- initialize weights
			params:copy(initParams)

			-- train 
			local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.test)
			for i=1,input:size(1)*optimOpt.nUpdate do
				local idx = math.fmod(i-1, input:size(1)) + 1
				
				-- evaluation network on current batch
				local function feval(x)

					-- zero-out gradients
					gParams:zero()

					-- get new parameters
					if x ~= params then
						params:copy(x)
					end

					-- evaluation network and loss
					local inputSel = input:index(1,torch.LongTensor{idx}) 
					local prediction = network:forward(inputSel)
					local loss = criterion:forward(prediction, target[idx])
					local dloss = criterion:backward(prediction, target[idx])
					network:backward(inputSel, dloss)

					return loss, gParams
				end

				-- update parameters
				blah, f = optim.sgd(feval, params, optimOptCopy)
			end

			-- test	
			local prediction = network:forward(testData.input)
			for i=1,prediction:size(1) do
				cM:add(prediction[i], testData.target[i])
			end
		end)	
	end
	
	local perf = {}
	return _.map(conf, function(k,cM)
            cM:updateValids()
				return cM.totalValid*100
          end)			
end

function bestSGD(model, nClasses, evalSet, conf, opt, learningRates, learningRateDecays, nUpdates)
	local network = model.net
	network:remove(network:size())
	network:add(nn.Linear(model.outSize, nClasses))
	model.net = util.localize(model.net, opt)

	local preTrainedParams, gParams = network:getParameters()
	preTrainedParams = preTrainedParams:clone()

	local bestPerf = {}
	_.map(conf, function(k,cM) bestPerf[k] = {params=0, accuracy=0} end)

	-- loop over variables to grid search over
	_.each(learningRates, function(i, lr)
		_.each(learningRateDecays, function(j, lrDecay)	
			_.each(nUpdates, function(u, update) 			
				
				-- update best performance on each task
				local optimOpt = {learningRate=lr, learningRateDecay=lrDecay, nUpdate=update}
				print("evaluating params: ")
				print(optimOpt)
				local kShotAccs = eval(network, model.criterion, evalSet, conf, preTrainedParams, opt, optimOpt)
				
				_.each(conf, function(k, cM)  
					print(k .. '-shot: ')
					print(cM)
					if kShotAccs[k] > bestPerf[k].accuracy then
						bestPerf[k].params = optimOpt
						bestPerf[k].accuracy = kShotAccs[k]
					end
				end)
			end)
		end)
	end)

	-- reset params back
	local params, gParams = network:getParameters()
	params:copy(preTrainedParams)

	return bestPerf
end

return function(opt, model, dataset, confusion)
	opt.bestSGD = bestSGD 
	return require(opt.homePath .. 'model.baselines.nearest-neighbor')(opt, model, dataset, confusion)	
end
