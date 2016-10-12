require 'cortex-core.projects.research.oneShotLSTM.model.lstm.lstm-helper'

local t = require 'torch'
local autograd = require 'autograd'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'

function getMetaLearner1(opt)
   local metaLearner = {}
   local nHidden = opt.nHidden or 20
   local maxGradNorm = opt.maxGradNorm or 0.25 

   local lstm1, params, layers = require(opt.homePath .. '.model.lstm.RecurrentLSTMNetwork')({  
      inputFeatures = 2,   -- preGrad(2)
      hiddenFeatures = nHidden,
      outputType = 'all',
      batchNormalization = opt.BN1, 
      maxBatchNormalizationLayers = opt.steps 
   })
   local lstm2, params2, layers2 = require(opt.homePath .. '.model.lstm.RecurrentLSTMNetwork')({ 
      inputFeatures = nHidden,
      hiddenFeatures = nHidden,
      outputType = 'last',
      batchNormalization = opt.BN2, 
      maxBatchNormalizationLayers = opt.steps,
   }, params, layers)

   for i, weights in pairs(params) do
      for j, weight in pairs(weights) do  
         weights[j]:uniform(-0.1,0.1)
      end 
   end

   local outputW = torch.rand(nHidden, 1)
   params[3] = outputW
   local cI = torch.rand(opt.nParams, 1)
   cI:uniform(-0.1, 0.1)
   params[4] = cI

   metaLearner.params = params
   metaLearner.layers = layers 
   
   metaLearner.forward = function(metaLearnerParams, layers, input, prevState)
      local preGrad = input[1]

      local h, newState1 = lstm1(metaLearnerParams[1], preGrad, prevState[1], layers[1])  
      local c, newState2 = lstm2(metaLearnerParams[2], h, prevState[2], layers[2])
      
      return c * metaLearnerParams[3], {newState1, newState2}
   end

   metaLearner.f = function(metaLearnerParams, learner, trainInput, trainTarget, testInput, testTarget, steps, batchSize)
   
      local metaLearnerState = {}
      local learnerParamsState = {}
      learnerParamsState[1] = metaLearnerParams[4] 

      local learnerParams = metaLearnerParams[4]   
      local trainSize = trainInput:size(1)      
      local idx = 1

      -- training set loop
      for s=1,steps do

         -- shuffle?
         if toShuffle then 
            local shuffle = torch.randperm(trainInput:size(1))
            trainInput = trainInput:index(1, shuffle:long())
            trainTarget = trainTarget:index(1, shuffle:long())
         end
         
         for i=1,trainSize,batchSize do 
            -- get image input & label
            local x = trainInput[{{i,i+batchSize-1},{},{},{}}]
            local y = trainTarget[{{i,i+batchSize-1}}]

            -- get gradient and loss w/r/t input+label      
            local gradLearner, lossLearner = learner.df(getValue(learnerParamsState[idx]), x, y)                        
            
            -- preprocess grad & loss 
            gradLearner = torch.view(gradLearner, gradLearner:size(1), 1, 1)
            local preGrad, preLoss = preprocess(gradLearner, lossLearner)

            -- use meta-learner to get learner's next parameters
            local state = metaLearnerState[idx-1] or {{},{}} 
            local cOut, sOut = metaLearner.forward(metaLearnerParams, layers, {preGrad}, state)    
            metaLearnerState[idx] = sOut 
            learnerParamsState[idx+1] = learnerParamsState[idx] + cOut

            -- break computational graph with getValue call 
            idx = idx + 1
         end
      end      
   
      -- Unflatten params and get loss+predictions from learner
      local learnerParamsFinal = learner.unflattenParams(learnerParamsState[#learnerParamsState])
      --local learnerParamsFinal = learner.unflattenParams(metaLearnerCell[#metaLearnerCell]) 
      return learner.f(learnerParamsFinal, testInput, testTarget)
   end
   
   metaLearner.df = autograd(metaLearner.f)  

   metaLearner.dfWithGradNorm = function(metaLearnerParams, learner, trainInput, trainTarget, testInput, testTarget, layers, prevState, steps)  
      local grads, loss, pred = metaLearner.df(metaLearnerParams, learner, trainInput, trainTarget, testInput, testTarget, layers, prevState, steps)  

      local norm = 0
      for i,grad in ipairs(autograd.util.sortedFlatten(grads)) do
         norm = norm + torch.sum(torch.pow(grad,2))
      end
      norm = math.sqrt(norm)
      if norm > maxGradNorm then
         for i,grad in ipairs(autograd.util.sortedFlatten(grads)) do
            grad:mul( maxGradNorm / norm )
         end
      end

      return grads, loss, pred   
   end

   return metaLearner
end

