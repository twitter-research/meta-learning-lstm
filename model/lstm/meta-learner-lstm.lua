require 'model.lstm.lstm-helper'

local t = require 'torch'
local autograd = require 'autograd'
local DirectNode = require 'autograd.runtime.direct.DirectNode'
local util = require 'util.util'

function getLearner(opt)
   local learner = {}
   local model = require(opt.model)({
      nClasses=opt.nClasses.train,
      classify=true,
      useCUDA=opt.useCUDA,
      nIn=opt.nIn,
      nDepth=opt.nDepth,
      BN_momentum=opt.BN_momentum
   }) 
   
   local netF, learnerParams = autograd.functionalize(model.net:clone('running_mean', 'running_var'))      
   learner.params = learnerParams
   local parameters, gradParameters = model.net:getParameters()
   
   learner.unflattenParams = function(flatParams)
      return util.unflattenParamsArb(learner.params, flatParams)
   end 
   
   learner.nParams = model.nParams  

   local loss, _ = autograd.functionalize(model.criterion:clone())
   learner.forward = function(params, input)
      local out = netF(params, input)
      
      return out
   end
   learner.f = function(params, input, testY)
      local out = learner.forward(params, input)
    
      return loss(out, testY), out 
   end  
   
   local feval = function(x, inputs, targets)
      if x ~= parameters then
         parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- evaluate function for complete mini batch
      local outputs = model.net:forward(inputs)
      local f = model.criterion:forward(outputs, targets)

      --print(targets)
      --print(softmax:forward(outputs))

      -- estimate df/dW
      local df_do = model.criterion:backward(outputs, targets)
      model.net:backward(inputs, df_do) 
      gradParameters:div(inputs:size(1))

      -- return f and df/dX
      return gradParameters:clone(), f
   end   

   -- needs to return grads, loss
   learner.df = function(params, input, testY)
      return feval(params, input, testY)     
   end 
 
   learner.reset = function()
      netF.module:reset()
      model.net:reset()
   end
   
   learner.set = function(mode)
      if mode == 'training' then
         netF.module:training()
         model.net:training()
      elseif mode == 'evaluate' then
         netF.module:evaluate()
         model.net:evaluate()
      else
         error(string.format("model.set: undefined mode - %s", mode))
      end
   end 


   return learner
end

function getMetaLearner2(opt)
   local metaLearner = {}
   local nHidden = opt.nHidden or 20
   local maxGradNorm = opt.maxGradNorm or 0.25 

   local nLstm, params, layers = require('model.lstm.RecurrentLSTMNetwork')({  
      inputFeatures = 4,   -- loss(2) + preGrad(2) = 4
      hiddenFeatures = nHidden,
      outputType = 'all',
      batchNormalization = opt.BN1, 
      maxBatchNormalizationLayers = opt.steps 
   })
   local mlLstm, params2, layers2 = getMetaLearnerLSTM({
      nParams = opt.nParams,
      nInput = nHidden,
      debug = opt.debug,
      batchNormalization = opt.BN2, 
      maxBatchNormalizationLayers = opt.steps
   }, params, layers)

   metaLearner.params = params
   metaLearner.layers = layers 

   -- initialize weights
   for i, weights in pairs(metaLearner.params) do
      for j, weight in pairs(weights) do 
         weights[j]:uniform(-0.01,0.01)
      end
   end

   --[[
      want initial forget value to be high and input value to be low so that
      model starts with gradient descent
   --]]
   metaLearner.params[2].bF:uniform(4,5)  --(1,2)  
   metaLearner.params[2].bI:uniform(-4,-5)

   -- set initial cell state = learner's initial parameters
   local initialParam = nn.Module.flatten(opt.learnerParams):clone()
   initialParam = torch.view(initialParam, initialParam:size(1), 1)
   metaLearner.params[2].cI:copy(initialParam)
   
   metaLearner.forward = function(metaLearnerParams, layers, input, prevState)
      local loss = input[1]
      local preGrad = input[2]
      local grad = input[3]

      local lossExpand = torch.expandAs(loss, preGrad)
      local input1 = torch.cat(lossExpand, preGrad, 3)
      local h, newState1, _ = nLstm(metaLearnerParams[1], input1, prevState[1], layers[1])
      local c, newState2 = mlLstm(metaLearnerParams[2], {h, grad}, prevState[2], layers[2])

      return c, {newState1, newState2}
   end 

   metaLearner.f = function(metaLearnerParams, learner, trainInput, trainTarget, testInput, testTarget, steps, batchSize, evaluate)
      -- set learner's initial parameters = inital cell state 
      local learnerParams = torch.clone(DirectNode.getValue(metaLearnerParams[2].cI)) 
   
      local metaLearnerState = {}
      local metaLearnerCell = {}

      local trainSize = trainInput:size(1)      
      local idx = 1

      learner.reset()
      learner.set('training')

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
            --local x = trainInput[{{i,i+batchSize-1},{},{},{}}]
            --local y = trainTarget[{{i,i+batchSize-1}}]
            local x,y = util.getBatch(trainInput, trainTarget, i, batchSize)

            -- get gradient and loss w/r/t input+label      
            local gradLearner, lossLearner = learner.df(learnerParams, x, y)                    
            --print(idx .. ": " .. lossLearner)

            -- preprocess grad & loss 
            gradLearner = torch.view(gradLearner, gradLearner:size(1), 1, 1)
            local preGrad, preLoss = preprocess(gradLearner, lossLearner)

            -- use meta-learner to get learner's next parameters
            local state = metaLearnerState[idx-1] or {{},{}} 
            local cOut, sOut = metaLearner.forward(metaLearnerParams, layers, {preLoss, preGrad, gradLearner}, state)   
            metaLearnerState[idx] = sOut 
            metaLearnerCell[idx] = cOut
            
            -- break computational graph with getValue call 
            learnerParams = DirectNode.getValue(cOut)
            idx = idx + 1
         end
      end      
   
      -- Unflatten params and get loss+predictions from learner
      local learnerParamsFinal = learner.unflattenParams(metaLearnerCell[#metaLearnerCell]) 
      if evaluate then learner.set('evaluate') end 
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

