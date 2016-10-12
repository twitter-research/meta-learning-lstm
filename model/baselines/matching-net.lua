local t = require 'torch'
local nn = require 'nn'
local autograd = require 'autograd'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
local _ = require 'moses'

function getMatchingNet(opt)
   local model = {}  

   -- load and functionalize cosine-similarity layer
   local cosineSim = autograd.functionalize(nn.CosineDistance())  
   
   -- load embedding model (simple or FCE)
   local embedModel = require(opt.homePath .. opt.embedModel)(opt)
   local cast = "float"
   if opt.useCUDA then
      cast = "cuda"
   end
   model.params = autograd.util.cast(embedModel.params, cast)        
   
   -- set training or evaluate function
   model.set = embedModel.set    

   -- trainInput is numTrain x 1 x 28 x 28
   -- trainTarget is numTrain x 1
   -- testInput is batchSize x 1 x 28 x 28
   model.forward = function(params, input)
      local trainInput = input[1]
      local trainTarget = input[2]
      local testInput = input[3]
      local batchSize = input[3]:size(1)
      
      -- embed train & test 
      local gS = embedModel.embedS(params, testInput)
      local fX = embedModel.embedX(params, trainInput, gS, 3)  
      
      local out = {}
      local y_one_hot = autograd.util.oneHot(trainTarget, opt.nClasses.train) 
      -- batch loop
      for i=1,batchSize do
         -- get cosine weights over train examples for each test example
         local weight = cosineSim({fX, t.expandAs(torch.reshape(gS[i], 1, gS[i]:size(1)), fX)})
         local output = torch.cmul(t.expandAs(t.reshape(weight, weight:size(1), 1), y_one_hot), y_one_hot)
         local sum = torch.sum(output, 1)

         -- apply log softmax
         local log_y_hat = sum - t.log(t.sum(t.exp(sum)))
         out[i] = log_y_hat
      end       

      return torch.cat(out, 1)   
   end

   model.f = function(params, input, testY)     
      local out = model.forward(params, input)
      local y_one_hot = autograd.util.oneHot(testY, opt.nClasses.train) 
      local loss = t.mean(-t.sum(t.cmul(out, y_one_hot),2))
      
      return loss, out 
   end

   model.df = embedModel.df(autograd(model.f))     

   return model
end

return function(opt, dataset) 
   local metaTrainSet = dataset.train
   local metaValidationSet = dataset.validation
   local metaTestSet = dataset.test

   -- model
   local network = getMatchingNet(opt) 
   print('params: ')
   print(network.params)

   -- keep track of errors
   local avgs = {} 
   local trainConf = optim.ConfusionMatrix(opt.nClasses.train)
   local valConf = {}
   local testConf = {}
   for _,k in pairs(opt.nTestShot) do
      valConf[k] = optim.ConfusionMatrix(opt.nClasses.val)
      testConf[k] = optim.ConfusionMatrix(opt.nClasses.test)
      avgs[k] = 0 
   end

   local nEpisode = opt.nEpisode 
   local printPer = opt.printPer 
   
   local timer = torch.Timer()
   local cost = 0 
   local optimState = {learningRate=opt.learningRate}
   local evalCounter = 1   

   -- init optimizer
   local optimizer, optimState = autograd.optim[opt.optimMethod](network.df, tablex.deepcopy(opt), network.params)
   
   -- set net for training 
   network.set('training')

   -- train episode loop   
   for d=1,nEpisode do
      local trainSet, testSet = metaTrainSet.createEpisode({testBatchSize=opt.batchSize}) 

      local nBatches = testSet:size()
      for epoch=1,opt.nEpochs do             
         -- get all train examples
         local trainData = trainSet:get()
      
         -- minibatch loop
         for idx=1,nBatches do   
            -- process test examples by batch
            local testData = testSet:get()
            local gParams, loss, prediction, _ = optimizer({trainData.input, trainData.target, testData.input}, testData.target)
            cost = cost + loss   
               
            -- update confusion matrix
            for i=1,prediction:size(1) do 
               trainConf:add(prediction[i], testData.target[i])
            end
   
         end
            
      end
      
      if math.fmod(d, opt.printPer) == 0 then
         local elapsed = timer:time().real   
         print('Training progress')
         print(string.format("Dataset: %d, Train Loss: %.3f, LR: %.5f, Time: %.4f s", d, cost/(printPer*nBatches), util.getCurrentLR(optimState[1]), elapsed))
         print(trainConf)
         trainConf:zero()
         
         -- evaluate validation set
         network.set('evaluate')
         for v=1,opt.nValidationEpisode do 
            local trainSet, testSet = metaValidationSet.createEpisode({testBatchSize=opt.batchSize})  
            
            -- get all train examples
            local trainData = trainSet:get() 
            
            -- k-shot loop
            for _,k in pairs(opt.nTestShot) do
               local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.val)
            
               -- minibatch loop
               local nBatches = testSet:size()  
               for idx=1,nBatches do  
                  -- process test example
                  local testData = testSet:get() 
                  local pred = network.forward(network.params, {input, target, testData.input})
                  
                  for i=1,pred:size(1) do  
                     valConf[k]:add(pred[i], testData.target[i])
                  end 
               end   
            end
         end   
      
         for _,k in pairs(opt.nTestShot) do 
            print('Validation Accuracy (' .. k .. '-shot)')
            print(valConf[k])
            valConf[k]:zero()
         end

         cost = 0
         timer = torch.Timer()
         network.set('training')
      end
   end   
   
   -- set net for testing
   network.set('evaluate')

   -- test episode loop 
   for d=1,opt.nTest do 
      local trainSet, testSet = metaTestSet.createEpisode({testBatchSize=opt.batchSize})   

      -- get all train examples
      local trainData = trainSet:get() 
   
      for _, k in pairs(opt.nTestShot) do 
         local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.test)
         
         -- minibatch loop
         local nBatches = testSet:size()  
         for idx=1,nBatches do 
            -- process test example
            local testData = testSet:get() 
            local pred = network.forward(network.params, {input, target, testData.input}) 
            for i=1,pred:size(1) do 
               testConf[k]:add(pred[i], testData.target[i])
            end
         end   
      end

   end   

   for _,k in pairs(opt.nTestShot) do 
      print('Test Accuracy (' .. k .. '-shot)')
      print(testConf[k])
   end
      
   return _.values(_.map(testConf, function(i,cM) 
            return i .. '-shot: ' .. cM.totalValid*100
          end))
end
