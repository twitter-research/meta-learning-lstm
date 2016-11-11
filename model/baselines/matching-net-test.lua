local t = require 'torch'
local nn = require 'nn'
local autograd = require 'autograd'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
local _ = require 'moses'

local tester = torch.Tester()
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}

function getMatchingNet(opt)
   local model = {}  

   -- load and functionalize cosine-similarity layer
   local cosineSim = autograd.functionalize(nn.CosineDistance())  
   
   -- load embedding model (simple or FCE)
   local embedModel = require(opt.homePath .. opt.embedModel)(opt)
   local cast = "double"
   if opt.useCUDA then
      cast = "cuda"
   end
   model.params = autograd.util.cast(embedModel.params, cast)        
   
   -- set training or evaluate function
   model.set = embedModel.set    

   model.save = embedModel.save
   model.load = embedModel.load

   local lossF, _ = autograd.functionalize(util.localize(nn.CrossEntropyCriterion(), opt))

   -- trainInput is numTrain x 1 x 28 x 28
   -- trainTarget is numTrain x 1
   -- testInput is batchSize x 1 x 28 x 28
   model.forward1 = function(params, input)
      local trainInput = input[1]
      local trainTarget = input[2]
      local testInput = input[3]
      local batchSize = torch.size(testInput,1)

      -- embed support set & test items using g and f respectively  
      local gS = embedModel.embedS(params, trainInput)
      local fX = embedModel.embedX(params, testInput, gS, 3)  
      
      local out = {}
      local y_one_hot = autograd.util.oneHot(trainTarget, opt.nClasses.train) 
      
      -- test example loop 
      for i=1,batchSize do
         -- get cosine similarity over train examples for test item 
         local X = torch.clone(fX[i])
         local S = torch.clone(gS)
         local temp_hot = torch.clone(y_one_hot)

         local trainExample = torch.reshape(X, 1, torch.size(X,1))
         local weight = cosineSim({S, t.expandAs(trainExample, S)})
         
         -- element-wise multiply cosine similarity by one-hot-label matrix and sum to get total similarity score for each label
         local output = torch.cmul(t.expandAs(t.reshape(weight, torch.size(weight, 1), 1), temp_hot), temp_hot)
         local sum = torch.sum(output, 1)

         -- apply log softmax
         --local log_y_hat = sum - t.log(t.sum(t.exp(sum)))
         --local log_y_hat = autograd.util.logSoftMax(sum)
         --out[i] = log_y_hat
         out[i] = sum
      end       

      return torch.cat(out, 1)   
   end

   model.forward2 = function(params, input)
      local trainInput = input[1]
      local trainTarget = input[2]
      local testInput = input[3]

      local y_one_hot = autograd.util.oneHot(trainTarget, opt.nClasses.train)

      -- embed support set & test items using g and f respectively  
      local gS = embedModel.embedS(params, trainInput)
      local fX = embedModel.embedX(params, testInput, gS, 3)

      -- repeat tensors so that can get cosine sims in one call
      local repeatgS = torch.repeatTensor(gS, torch.size(fX, 1), 1)
      local repeatfX = torch.reshape(torch.repeatTensor(fX, 1, torch.size(gS, 1)), 
         torch.size(fX, 1)*torch.size(gS,1), torch.size(fX, 2))
      
      -- weights are num_test x num_train (weights per test item)
      local weights = torch.reshape(cosineSim({repeatgS, repeatfX}), torch.size(fX, 1), torch.size(gS, 1), 1)
      
      -- one-hot matrix of train labels is expanded to num_train x num_test x num_labels
      local expandOneHot = torch.expand(torch.reshape(y_one_hot, 1, torch.size(y_one_hot, 1), 
         torch.size(y_one_hot, 2)), torch.size(fX, 1), torch.size(y_one_hot, 1), torch.size(y_one_hot, 2))
      
      -- weights are expanded to match one-hot matrix
      local expandWeights = torch.expandAs(weights, expandOneHot)

      -- cmul one-hot matrix by weights and sum along rows to get weight per label
      -- final size: num_train x num_labels 
      local out = torch.reshape(torch.sum(torch.cmul(expandWeights, expandOneHot), 2), 
         torch.size(fX, 1), torch.size(y_one_hot, 2))

      return out 
   end

   model.f1 = function(params, input, testY)    
      local lossT = {}
      local outT = {}
      for i=1,#input do  
         local out = model.forward1(params, input[i])     
         
         local loss = lossF(out, testY[i])
         lossT[i] = loss 
         outT[i] = out
      end

      return torch.mean(autograd.util.cat(lossT)), torch.cat(outT, 1) 
   end

   model.f2 = function(params, input, testY)    
      local lossT = {}
      local outT = {}
      for i=1,#input do  
         local out = model.forward2(params, input[i])     
         
         local loss = lossF(out, testY[i])
         lossT[i] = loss 
         outT[i] = out
      end

      return torch.mean(autograd.util.cat(lossT)), torch.cat(outT, 1) 
   end


   model.df1 = embedModel.df(autograd(model.f1))     
   model.df2 = embedModel.df(autograd(model.f2))

   return model
end

opt = {}
opt.task = 'config.5-shot-5-class' 
opt.data = 'config.omniglot'
opt.model = 'config.baselines.train-matching-net'

-- load data
-- load config info for task, data, and model 
opt = require('cortex-core.projects.research.oneShotLSTM.' .. opt.task)(opt)
opt = require('cortex-core.projects.research.oneShotLSTM.' .. opt.data)(opt)
opt = require('cortex-core.projects.research.oneShotLSTM.' .. opt.model)(opt)

train = require(opt.homePath .. opt.dataLoader)(opt, opt.trainFile)  
local layer = 1 

function getFiniteDiff(fxn, params, batchInput, batchTarget)
   local eps = 1e-6

   local temp = params[layer][1][1][1][1]
   params[layer][1][1][1][1] = temp + eps
   local loss1, _ = fxn(params, batchInput, batchTarget)
   params[layer][1][1][1][1] = temp - eps
   local loss2, _ = fxn(params, batchInput, batchTarget)

   params[layer][1][1][1][1] = temp
   return (loss1 - loss2)/(2*eps)
end

-- load network
opt.useDOUBLE = true
local network = getMatchingNet(opt) 

local batchInput = {}
local batchTarget = {}

local trainSet, testSet = train.createEpisode({testBatchSize=2})

-- get all train examples
local trainData = trainSet:get()
local testData = testSet:get()

table.insert(batchInput, autograd.util.cast({trainData.input, trainData.target, testData.input}, "double"))
table.insert(batchTarget, autograd.util.cast(testData.target, "double"))

print(batchInput)
print(batchTarget)
print(network.params)

print('actual1: ' .. getFiniteDiff(network.f1, network.params, batchInput, batchTarget))
--tester:assert(gradcheck(network.f2, network.params, autograd.util.cast(batchInput, "double"), autograd.util.cast(batchTarget, "double")), 'incorrect gradients for (2)')
--tester:assert(gradcheck(network.f1, network.params, batchInput, batchTarget), 'incorrect gradients for (1)')

-- method1
local df1, loss1, out1 = network.df1(network.params, batchInput, batchTarget)
print(loss1)
print(out1)

print('calc1: ' .. df1[layer][1][1][1][1])

--[[
-- method1 repeated
local loss1, out1 = network.f1(network.params, batchInput, batchTarget)
local df1 = network.df1(network.params, batchInput, batchTarget)
print(loss1)
print(out1)
for i=1,10 do
   print(i .. ": ")
   print(torch.sum(df1[i]))
end
--]]

print('')
print('actual2: ' .. getFiniteDiff(network.f2, network.params, batchInput, batchTarget))
-- method2
local loss2, out2 = network.f2(network.params, batchInput, batchTarget)
print(loss2)
print(out2)
--local df2 = network.df2(network.params, batchInput, batchTarget)

--print('calc2: ' .. df2[layer][1][1][1][1])
