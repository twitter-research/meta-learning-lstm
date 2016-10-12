local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
local nearestNeighborLib = require 'cortex-core.projects.research.oneShotLSTM.model.baselines.nearest-neighbor'

return function(opt, dataset) 
   -- dataset
   local metaTrainSet = dataset.train
   local metaValidationSet = dataset.validation
   local metaTestSet = dataset.test 
   
   -- model
   local model = require(opt.homePath .. opt.model) ({
      nClasses = opt.nAllClasses, 
      useCUDA = opt.useCUDA,
      classify = true,
      nIn = opt.nIn,
      nDepth = opt.nDepth  
   })
   local network = model.net
   local criterion = model.criterion

   -- keep track of errors
   local avgs = {}  
   local trainConf = optim.ConfusionMatrix(opt.nAllClasses)
   local valConf = {}
   local testConf = {}
   for _,k in pairs(opt.nTestShot) do
      valConf[k] = optim.ConfusionMatrix(opt.nClasses.val)
      testConf[k] = optim.ConfusionMatrix(opt.nClasses.test)
      avgs[k] = 0 
   end   

   local cost = 0    
   
   -- train network only if using convolutional embedding
   if opt.convNearestNeighbor or opt.preTrainSGD then    
      -- set up training
      local params, gParams = network:getParameters()  
      local optimOpt = {learningRate=opt.learningRate, learningRateDecay=opt.learningRateDecay}

      -- epoch loop
      for epoch=1,opt.nEpochs do    
         -- create episode with all classes 
         local trainSet, testSet = metaTrainSet.createEpisode({trainBatchSize=opt.batchSize})
         local nBatches = trainSet:size() 

         -- minibatch loop
         for idx=1,nBatches do

            -- get minibatch
            local data = trainSet:get()

            -- evaluation network on current batch
            local function feval(x)
               
               -- zero-out gradients
               gParams:zero()

               -- get new parameters
               if x ~= params then
                  params:copy(x)
               end

               -- evaluation network and loss
               local prediction = network:forward(data.input)
               local loss = criterion:forward(prediction, data.target)
               local dloss = criterion:backward(prediction, data.target)
               network:backward(data.input, dloss)

               -- update confusion matrix
               for i=1,prediction:size(1) do
                  trainConf:add(prediction[i], data.target[i])
               end

               return loss, gParams
            end   
            
            -- update parameters
            local _, f = optim.sgd(feval, params, optimOpt)
            cost = cost + f[1]   
         end
         
         if math.fmod(epoch, opt.printPer) == 0 then
            trainConf:updateValids()   
            print(string.format("Training epoch: %d, Train Loss: %.3f, Accuracy: %.3f", epoch, cost/(opt.printPer * nBatches), trainConf.totalValid*100))
            trainConf:zero()

            if opt.convNearestNeighbor then 
               -- evaluate validation set
               for v=1,opt.nValidationEpisode do
                  
                  --print('metaValidation Train')
                  local trainSet, testSet = metaValidationSet.createEpisode({})
         
                  -- get all train and test examples
                  local trainData = trainSet:get()
                  local testData = testSet:get()   
      
                  -- k-shot loop
                  for _,k in pairs(opt.nTestShot) do
                     local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.val)
      
                     local pred = nearestNeighborLib.classify(network, input, target, testData.input, opt)
                     for i=1,pred:size(1) do
                        valConf[k]:add(pred[i], testData.target[i])
                     end

                  end
               end

               for _,k in pairs(opt.nTestShot) do
                  print('Validation Accuracy (' .. k .. '-shot)')
                  print(valConf[k])
                  valConf[k]:zero()
               end
            end   
            
            cost = 0
            network:training()   
         end
      end
   end

   if opt.convNearestNeighbor or opt.pixelNearestNeighbor then 
      -- test loop
      for d=1,opt.nTest do
         -- create a new episode for each test 
         local trainSet, testSet = metaTestSet.createEpisode({})   

         -- get all train & test 
         local trainData = trainSet:get() 
         local testData = testSet:get() 

         for _, k in pairs(opt.nTestShot) do
            local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.test)

            local pred = nearestNeighborLib.classify(network, input, target, testData.input, opt)
            for i=1,#pred do
               testConf[k]:add(pred[i], testData.target[i])
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
   elseif opt.preTrainSGD then
      
      -- find best hyper-parameters on validation set 
      local bestPerf = opt.bestSGD(model, opt.nClasses.val, metaValidationSet, valConf, opt, opt.learningRates, opt.learningRateDecays, opt.nUpdates)    
      print('Best params: ')
      print(bestPerf[opt.nTrainShot].params)
      local lr, lrDecay = bestPerf[opt.nTrainShot].params.learningRate, bestPerf[opt.nTrainShot].params.learningRateDecay
      local nUpdate = bestPerf[opt.nTrainShot].params.nUpdate

      -- evaluate best hyper-parameters on test set
      local perf = opt.bestSGD(model, opt.nClasses.test, metaTestSet, testConf, opt, {lr}, {lrDecay}, {nUpdate})
      print(perf)
      return _.values(_.map(perf, function(k, p)
               return k .. '-shot: ' .. p.accuracy 
            end)) 
   end   
end
