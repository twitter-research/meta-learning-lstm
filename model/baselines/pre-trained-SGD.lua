local util = require 'util.util'
local _ = require 'moses'

function eval(savedNetwork, criterion, evalSet, nEpisodes, conf, opt, optimOpt) 
   local acc = {}
   _.each(conf, function(k, cM)  
      acc[k] = torch.zeros(nEpisodes)
   end)
   
   -- evaluate validation set
   for v=1,nEpisodes do
      local trainSet, testSet = evalSet.createEpisode({})
   
      -- get all train examples
      local trainData = trainSet:get()
      local testData = testSet:get()

      -- k-shot test loop
      _.each(conf, function(k, cM)
         local optimOptCopy = util.deepClone(optimOpt)
   
         -- load initial network to use 
         local network = savedNetwork:clone()
         network:training()

         local params, gParams = network:getParameters() 

         -- train 
         local input, target = util.extractK(trainData.input, trainData.target, k, opt.nClasses.test)
         for i=1,optimOpt.nUpdate do
            -- evaluation network on current batch
            local function feval(x)
               -- zero-out gradients
               gParams:zero()

               -- get new parameters
               if x ~= params then
                  params:copy(x)
               end

               -- evaluation network and loss
               local prediction = network:forward(input)
               local loss = criterion:forward(prediction, target)
               local dloss = criterion:backward(prediction, target)
               network:backward(input, dloss)

               return loss, gParams
            end

            -- update parameters
            blah, f = optim.sgd(feval, params, optimOptCopy)
         end

         -- test  
         network:evaluate()
         local prediction = network:forward(testData.input)
         for i=1,prediction:size(1) do
            cM:add(prediction[i], testData.target[i])
         end
         cM:updateValids()
         acc[k][v] = cM.totalValid * 100
         cM:zero() 
      end)  
   end
   
   return acc 
end

function bestSGD(model, nClasses, evalSet, nEpisodes, conf, opt, learningRates, learningRateDecays, nUpdates)
   -- replace last linear layer with new layer
   model.net:remove(model.net:size())
   model.net:add(nn.Linear(model.outSize, nClasses))
   model.net = util.localize(model.net, opt)
 
   local savedNetwork = model.net:clone() 
   local bestPerf = {}
   _.map(conf, function(k,cM) bestPerf[k] = {params=0, accuracy=0, accVector} end)

   -- loop over hyperparameters to grid search over
   _.each(learningRates, function(i, lr)
      _.each(learningRateDecays, function(j, lrDecay) 
         _.each(nUpdates, function(m, update)    
            
            -- update best performance on each task
            local optimOpt = {learningRate=lr, learningRateDecay=lrDecay, nUpdate=update}
            print("evaluating params: ")
            print(optimOpt)
            
            local kShotAccs = eval(savedNetwork, model.criterion, evalSet, nEpisodes, conf, opt, optimOpt)
            _.each(kShotAccs, function(k, acc)  
               print(k .. '-shot: ')
               print(acc:mean())
               if acc:mean() > bestPerf[k].accuracy then
                  bestPerf[k].params = optimOpt
                  bestPerf[k].accuracy = acc:mean()
                  bestPerf[k].accVector = acc 
               end
            end)
         end)
      end)
   end) 

   return bestPerf
end

return function(opt, dataset)
   opt.bestSGD = bestSGD 
   return require('model.baselines.pre-train')(opt, dataset) 
end
