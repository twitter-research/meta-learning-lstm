local t = require 'torch'
local autograd = require 'autograd'
local funcNN = autograd.functionalize('nn') 
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'

local P = 10
local expP = torch.exp(P)
local negExpP = torch.exp(-P)

function preProc1(x)
	local absX = t.abs(x) 
	local cond1 = torch.gt(absX, negExpP)
   local cond2 = torch.le(absX, negExpP)

   local x1 = x:maskedSelect(cond1)
   x1 = t.log(t.abs(x1))/P
   local x2 = x:maskedSelect(cond2)
   x2:fill(-1) 

   local z = util.zerosAs(x, x:size())
	z:maskedCopy(cond1, x1) 
   z:maskedCopy(cond2, x2) 

   return z 
end

function preProc2(x)
	local absX = t.abs(x)
	local cond1 = torch.gt(absX, negExpP)
   local cond2 = torch.le(absX, negExpP)

   local x1 = x:maskedSelect(cond1)
   x1 = t.sign(x1) 
   local x2 = x:maskedSelect(cond2)
   x2 = x2*expP

   local z = util.zerosAs(x, x:size())
	z:maskedCopy(cond1, x1) 
   z:maskedCopy(cond2, x2) 

   return z 
end

-- pre-processing according to Deepmind 'Learning to Learn' paper
function preprocess(grad, loss)
	local preGrad = torch.zero(grad.new(grad:size(1), 1, 2))
	preGrad[{{},{},{1}}] = preProc1(grad)
	preGrad[{{},{},{2}}] = preProc2(grad)	

	local lossT = util.zerosAs(grad, torch.LongStorage({1,1,1}))
	lossT[1] = loss	
	local preLoss = util.zerosAs(grad, torch.LongStorage({1,1,2}))
	preLoss[{{},{},{1}}] = preProc1(lossT) 
	preLoss[{{},{},{2}}] = preProc2(lossT)

	return preGrad, preLoss 
end

--[[function getMetaLearnerInput(lopt) 
	local metaLearner = lopt.metaLearner
	local metaLearnerState = lopt.metaLearnerState
	local steps = lopt.steps or trainSize
	
	local learner = lopt.learner
	local input = lopt.input
	local target = lopt.target

	local trainSize = input:size(1)
	local preGrads = {}
	local grads = {}
	local losses = {}		

	-- set learner's initial parameters = inital cell state 
	learner.params = learner.unflattenParams(metaLearner.params[2].cI)	
	local learnerParams = learner.params	

	local flatParams	
	local grad_list = {}
	local loss_list = {}
	-- training set loop
	for s=0,steps-1 do	
		local getTimer = torch.Timer()	
		-- get image input & label
		local x = input[{{math.fmod(s,trainSize)+1},{},{},{}}]
		local y = target[{{math.fmod(s,trainSize)+1}}]

		-- get gradient and loss w/r/t input+label	
		local grad, loss, pred = learner.df(learnerParams, x, y) 				 		

		-- nan check
		local gradNan = util.NaNCheck(grad) 
		if #gradNan > 0  or loss ~= loss then
			print('NaN at step ' .. s .. '. Stopping...')
			print('Grad check')
			print(gradNan)
			print('loss: ' .. loss)

			print('loss list')
			print(loss_list)
			print('grad list')
			print(grad_list)
			break
		else
			table.insert(loss_list, loss)
			table.insert(grad_list, util.paramNorm(grad))
		end
	
		-- preprocess grad, loss inputs 
		grad = nn.Module.flatten(grad)	
		grad = torch.view(grad, grad:size(1), 1, 1)
		local lossT = torch.zero(grad.new(1,1,1))
		lossT[1] = loss
		local preGrad, preLoss = preprocess(grad, loss)

		-- use meta-learner to get learner's next parameters
		flatParams, metaLearnerState = metaLearner.forward(metaLearner.params, metaLearner.layers, {preLoss, preGrad, grad}, metaLearnerState)

		-- convert flattened params to format of learner 
		learnerParams = learner.unflattenParams(flatParams)

		-- record gradient and loss 
		losses[s] = preLoss 
		preGrads[s] = preGrad 
		grads[s] = grad 
	end
	
	return {torch.cat(losses,2), torch.cat(preGrads, 2), torch.cat(grads, 2)}, loss_list, grad_list
end
--]]
---------------------------------------------------
function getMetaLearnerLSTM(opt, params, layers)	
 	
   opt = opt or {}
   local nParams = opt.nParams 
	local m = 0 
	local nInput = opt.nInput 
	local batchNormalization = opt.batchNormalization or false
	local maxBatchNormalizationLayers = opt.maxBatchNormalizationLayers or 10

	local params = params or {}
	local layers = layers or {}
	local l = {}

	-- params
	local p = {
		WF = torch.zeros(nInput+2,1),
		WI = torch.zeros(nInput+2,1),
		cI = torch.zeros(nParams,1), 	-- initial cell state is a param 
		bI = torch.zeros(1,1),
		bF = torch.zeros(1,1)
	}

	if batchNormalization then 
		-- translation and scaling parameters are shared across time.
      local lstm_bn1, p_lstm_bn1 = funcNN.BatchNormalization(1)
      local lstm_bn2, p_lstm_bn2 = funcNN.BatchNormalization(1)

      l.lstm_bn1 = {lstm_bn1}
      l.lstm_bn2 = {lstm_bn2}	
		
		for i=2,maxBatchNormalizationLayers do 
			local lstm_bn1 = funcNN.BatchNormalization(1)
			local lstm_bn2 = funcNN.BatchNormalization(1)
			l.lstm_bn1[i] = lstm_bn1
			l.lstm_bn2[i] = lstm_bn2
		end

		-- initializing scaling to < 1 is recommended for LSTM batch norm.
      p.lstm_bn1_1 = p_lstm_bn1[1]:fill(0.1)
      p.lstm_bn1_2 = p_lstm_bn1[2]:zero()
      p.lstm_bn2_1 = p_lstm_bn2[1]:fill(0.1)
      p.lstm_bn2_2 = p_lstm_bn2[2]:zero()
		table.insert(layers, l)		
	end

	table.insert(params, p)

	-- function:
	-- x is table of {loss, gradient}
   local f = function(params, input, prevState, layers)	
		local x_all = input[1]
		local grad_input = input[2]
	
		local batch = t.size(grad_input, 1)
		local steps = t.size(grad_input, 2) 

		-- hiddens
		prevState = prevState or {}
		local fS = {}
		local iS = {}
		local cS = {}
		local deltaS = {}

		-- go over time steps of input
		for s=1,steps do
			-- loss, gradient inputs
			local x = t.select(x_all, 2, s)
			local act_g = t.select(grad_input, 2, s)	
		
			-- take care of BN
			local lstm_bn1, lstm_bn2
			if batchNormalization then 
				if layers.lstm_bn1[t] then
					lstm_bn1 = layers.lstm_bn1[t]
					lstm_bn2 = layers.lstm_bn2[t]
				else
					lstm_bn1 = layers.lstm_bn1[#layers.lstm_bn1]
					lstm_bn2 = layers.lstm_bn2[#layers.lstm_bn2]
				end
			end

			-- prev f, i, and c value
			local fP = fS[s-1] or prevState.f or torch.zero(grad_input.new(batch,1)) 
			local iP = iS[s-1] or prevState.i or torch.zero(grad_input.new(batch,1))
			local cP = cS[s-1] or prevState.c or params.cI  
			local deltaP = deltaS[s-1] or prevState.delta or torch.zero(grad_input.new(batch,1)) 

			-- next forget, input gate
			local fH = t.cat(cP, fP ,2)
			local iH = t.cat(cP, iP, 2)
			local FN, iN 
			if batchNormalization then 
				fN = lstm_bn1({params.lstm_bn1_1, params.lstm_bn1_2}, t.cat(x, fH, 2) * params.WF) + t.expand(params.bF, batch, 1)
				iN = lstm_bn2({params.lstm_bn2_1, params.lstm_bn2_2}, t.cat(x, iH, 2) * params.WI) + t.expand(params.bI, batch, 1)
			else
				fN = t.cat(x, fH, 2) * params.WF + t.expand(params.bF, batch, 1)
				iN = t.cat(x, iH ,2) * params.WI + t.expand(params.bI, batch, 1)
			end
			fS[s] = fN	
			iS[s] = iN
		
			-- next delta
			local delta = m * deltaP - t.cmul(autograd.util.sigmoid(t.view(iN,nParams,1)), act_g)

			-- next cell/params
			cS[s] = t.cmul(autograd.util.sigmoid(t.view(fN,nParams,1)), cP) + delta 
			
		end

		-- save state
		local newState = {f=fS[#fS], i=iS[#iS], c=cS[#cS]}
		-- return last cell
		return cS[#cS], newState	
	end

	return f, params, layers
end
