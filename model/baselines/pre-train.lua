local util = require 'util.util'
local nearestNeighborLib = require 'model.baselines.nearest-neighbor'
local autograd = require 'autograd'

return function(opt, dataset) 
   -- dataset
   local metaTrainSet = dataset.train
   local metaValidationSet = dataset.validation
   local metaTestSet = dataset.test 
   
   -- model
   local model = require(opt.model) ({
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
   
   -- train network only if using convolutional embedding or pre-train SGD
   if opt.convNearestNeighbor or opt.preTrainSGD then    
      -- set up training
      local params, gParams = network:getParameters()  
      
      -- load params from file?
      if opt.paramsFile then
         print('loading from: ' .. opt.paramsFile)
         local loadedModel = torch.load(opt.paramsFile)
         print(loadedModel)
         network = loadedModel
         model.net = network
         --params:copy(loadedParams) 
      end  
      
      local optimOpt = {learningRate=opt.learningRate}
      network:training()

      -- epoch loop
      for epoch=1,opt.nEpochs do    
         -- create episode with all classes 
         local trainSet, testSet = metaTrainSet.createEpisode({})
         local data = trainSet:get()   
         
         --print(data.target)
         --local input, target = util.extracKSpecial(data.input, data.target, 1, opt.nClasses.train) 
         local input, target = util.getRandomSubset(data.input, data.target, opt.nClasses.train) 

         -- evaluation network on current batch
         local function feval(x)
            
            -- zero-out gradients
            gParams:zero()

            -- get new parameters
            if x ~= params then
               params:copy(x)
            end

            --print(target)

            -- evaluation network and loss
            local prediction = network:forward(input)
            local loss = criterion:forward(prediction, target)
            local dloss = criterion:backward(prediction, target)
            network:backward(input, dloss) 

            -- update confusion matrix
            for i=1,prediction:size(1) do
               trainConf:add(prediction[i], target[i])
            end

            return loss, gParams
         end   
         
         -- update parameters
         local _, f = optim.adam(feval, params, optimOpt)
         cost = cost + f[1]   
         
         if math.fmod(epoch, opt.printPer) == 0 then
            trainConf:updateValids()   
            print(string.format("Training epoch: %d, Train Loss: %.3f, Accuracy: %.3f", epoch, cost/(opt.printPer), trainConf.totalValid*100))
            --print(string.format("Training epoch: %d, Train Loss: %.3f, Accuracy: %.3f", epoch, cost/(opt.printPer * nBatches), trainConf.totalValid*100))
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

            if math.fmod(epoch, 1000) == 0 then
               torch.save("conv-nearest-neighbor-model.th", network:clearState())
               --local prevIterParams = params:clone() 
               --torch.save("nearest-neighbor-baseline_params_snapshot.th", autograd.util.cast(prevIterParams, "float"))
            end 
         end
      end
   end

   local ret = {}
   if opt.convNearestNeighbor or opt.pixelNearestNeighbor then 
      -- test loop
      _.each(opt.nTest, function(i, n)
         local acc = {}
         for _, k in pairs(opt.nTestShot) do
            acc[k] = torch.zeros(n)
         end
 
         for d=1,n do
            -- create a new episode for each test 
            local trainSet, testSet = metaTestSet.createEpisode({})   

            -- get all train & test 
            local trainData = trainSet:get() 
            local testData = testSet:get() 

            for _, k in pairs(opt.nTestShot) do
               local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.test)

               local pred = nearestNeighborLib.classify(network, input, target, testData.input, opt)
               for i=1,pred:size(1) do
                  testConf[k]:add(pred[i], testData.target[i])
               end   
            
               testConf[k]:updateValids()
               acc[k][d] = testConf[k].totalValid*100
               testConf[k]:zero() 
            end
         
         end
                 
         for _,k in pairs(opt.nTestShot) do
            print('Test Accuracy (' .. n .. ' episodes, ' .. k .. '-shot)')
            --print(testConf[k])
            print(acc[k]:mean())
         end
  
         ret[n] = _.values(_.map(acc, function(i, val)
            local low = val:mean() - 1.96*(val:std()/math.sqrt(val:size(1)))
            local high = val:mean() + 1.96*(val:std()/math.sqrt(val:size(1)))
            return i .. '-shot: ' .. val:mean() .. '; ' .. val:std() .. '; [' .. low .. ',' .. high .. ']'
         end)) 
      end)

      return ret
   elseif opt.preTrainSGD then 
      -- find best hyper-parameters on validation set 
      local bestPerf = opt.bestSGD(model, opt.nClasses.val, metaValidationSet, opt.nValidationEpisode, valConf, opt, opt.learningRates, opt.learningRateDecays, opt.nUpdates)    
      print('Best params: ')
      print(bestPerf[opt.nTrainShot].params)
      local lr, lrDecay = bestPerf[opt.nTrainShot].params.learningRate, bestPerf[opt.nTrainShot].params.learningRateDecay
      local nUpdate = bestPerf[opt.nTrainShot].params.nUpdate

      -- evaluate best hyper-parameters on test set
      local perfArr = opt.bestSGD(model, opt.nClasses.test, metaTestSet, opt.nTest[#opt.nTest], testConf, opt, {lr}, {lrDecay}, {nUpdate})
      
      ret[opt.nTest[#opt.nTest]] = _.values(_.map(perfArr, function(i, perf)
         local val = perf.accVector
         local low = val:mean() - 1.96*(val:std()/math.sqrt(val:size(1)))
         local high = val:mean() + 1.96*(val:std()/math.sqrt(val:size(1)))
         return i .. '-shot: ' .. val:mean() .. '; ' .. val:std() .. '; [' .. low .. ',' .. high .. ']'
      end))
      
      return ret 
   end   
end
