local t = require 'torch'
local nn = require 'nn'
local autograd = require 'autograd'
local util = require 'util.util'
local _ = require 'moses'

function getMatchingNet(opt)
   local model = {}  

   -- load and functionalize cosine-similarity layer
   local cosineSim = autograd.functionalize(nn.CosineDistance())  
   
   -- load embedding model (simple or FCE)
   local embedModel = require(opt.embedModel)(opt)
   local cast = "float"
   if opt.useCUDA then
      cast = "cuda"
   end
   model.params = autograd.util.cast(embedModel.params, cast)        
   
   -- set training or evaluate function
   model.set = embedModel.set    

   model.save = embedModel.save
   model.load = embedModel.load

   local lossF, _ = autograd.functionalize(util.localize(
      nn.CrossEntropyCriterion(), opt))
 
   model.forward = function(params, input)
      local trainInput = input[1]
      local trainTarget = input[2]
      local testInput = input[3]
      local batchSize = input[3]:size(1)

      local y_one_hot = autograd.util.oneHot(trainTarget, opt.nClasses.train)

      -- embed support set & test items using g and f respectively  
      local gS = embedModel.embedS(params, trainInput)
      local fX = embedModel.embedX(params, testInput, gS, 3)

      -- repeat tensors so that can get cosine sims in one call
      local repeatgS = torch.repeatTensor(gS, torch.size(fX, 1), 1)
      local repeatfX = torch.reshape(torch.repeatTensor(fX, 1, torch.size(gS, 1)), 
         torch.size(fX, 1)*torch.size(gS,1), torch.size(fX, 2))
      
      -- weights are num_test x num_train (weights per test item)
      local weights = torch.reshape(cosineSim({repeatgS, repeatfX}), 
         torch.size(fX, 1), torch.size(gS, 1), 1)
      
      -- one-hot matrix of train labels is expanded to num_train x num_test x num_labels
      local expandOneHot = torch.expand(torch.reshape(y_one_hot, 1, 
         torch.size(y_one_hot, 1), torch.size(y_one_hot, 2)), 
         fX:size(1), torch.size(y_one_hot, 1), torch.size(y_one_hot, 2))
      
      -- weights are expanded to match one-hot matrix
      local expandWeights = torch.expandAs(weights, expandOneHot)

      -- cmul one-hot matrix by weights and sum along rows to get weight per label
      -- final size: num_train x num_labels 
      local out = torch.reshape(torch.sum(
         torch.cmul(expandWeights, expandOneHot), 2), 
         torch.size(fX, 1), torch.size(y_one_hot, 2))

      return out 
   end

   model.f = function(params, input, testY)    
      local lossT = {}
      local outT = {}
      for i=1,#input do  
         -- model's log-sofmax output 
         local out = model.forward(params, input[i])     
        
         -- calculate NLL 
         local loss = lossF(out, testY[i])
         lossT[i] = loss 
         outT[i] = out
      end

      return torch.mean(autograd.util.cat(lossT)), torch.cat(outT, 1) 
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

   -- load params from file?
   if opt.paramsFile and opt.networkFile then
      print('loading from params: ' .. opt.paramsFile)
      network.params = torch.load(opt.paramsFile)
      
      local cast = "float"
      if opt.useCUDA then
         cast = "cuda"
      end
      network.params = autograd.util.cast(network.params, cast)
   
      print('loading from network: ' .. opt.networkFile)
      network.load(opt.networkFile, opt) 
   end

   local nEpisode = opt.nEpisode 
   local printPer = opt.printPer 
   
   local timer = torch.Timer()
   local cost = 0 
   local optimState = {learningRate=opt.learningRate}
   local evalCounter = 1   

   ----------------------------------------------------------------------------
   -- meta-training 

   -- init optimizer
   local optimizer, optimState = autograd.optim[opt.optimMethod](network.df, 
      tablex.deepcopy(opt), network.params)
   
   -- set net for training 
   network.set('training')

   -- train episode loop   
   for d=1,nEpisode do

      local batchInput = {}
      local batchTarget = {}             
  
      -- get all train examples
      local trainSet, testSet = metaTrainSet.createEpisode(
         {testBatchSize=opt.batchSize})  
      local trainData = trainSet:get()
      local testData = testSet:get()   
      table.insert(batchInput, {trainData.input, trainData.target, testData.input})
      table.insert(batchTarget, testData.target)            
     
      local gParams, loss, prediction, _ = optimizer(batchInput, batchTarget)
      cost = cost + loss   
               
      -- update confusion matrix
      local idx = 1
      for i=1,batchTarget[1]:size(1) do 
         trainConf:add(prediction[idx], batchTarget[1][i])
         idx = idx + 1
      end
 
      if math.fmod(d, opt.printPer) == 0 then
         local elapsed = timer:time().real   
         print('Training progress')
         print(
            string.format("Dataset: %d, Train Loss: %.3f, LR: %.5f, Time: %.4f s", 
            d, cost/(printPer), util.getCurrentLR(optimState[1]), elapsed))
         print(trainConf) 
         trainConf:zero()
         
         -- evaluate validation set
         network.set('evaluate')
         for v=1,opt.nValidationEpisode do 
            local trainSet, testSet = metaValidationSet.createEpisode(
               {testBatchSize=opt.batchSize})  
            
            -- get all train examples
            local trainData = trainSet:get() 
            
            -- k-shot loop
            for _,k in pairs(opt.nTestShot) do
               local input, target = util.extractK(trainData.input, 
                  trainData.target, k, opt.nClasses.val)
            
               -- minibatch loop
               local nBatches = testSet:size()  
               for idx=1,nBatches do  
                  -- process test example
                  local testData = testSet:get() 
                  local pred = network.forward(network.params, {input, target, 
                     testData.input})
                  
                  for i=1,pred:size(1) do  
                     valConf[k]:add(pred[i], testData.target[i])
                  end 
               end   
            end
         end   
      
         for _,k in pairs(opt.nTestShot) do 
            print('Validation Accuracy (' .. opt.nValidationEpisode .. 
               ' episodes, ' .. k .. '-shot)') 
            print(valConf[k]) 
            valConf[k]:zero()
         end

         cost = 0
         timer = torch.Timer()
         network.set('training')
      end

      if math.fmod(d, 1000) == 0 then
         local prevIterParams = util.deepClone(network.params)
         torch.save("matching-net_params_snapshot.th", 
            autograd.util.cast(prevIterParams, "float"))
         network.save() 
      end
   end   
  
   ---------------------------------------------------------------------------- 
   -- meta-testing 
   
   -- set net for testing
   network.set('evaluate')

   local ret = {}
   _.each(opt.nTest, function(i, n)
      local acc = {}
      for _, k in pairs(opt.nTestShot) do
         acc[k] = torch.zeros(n)
      end
      
      for d=1,n do 
         local trainSet, testSet = metaTestSet.createEpisode(
            {testBatchSize=opt.batchSize})   

         -- get all train examples
         local trainData = trainSet:get() 
      
         for _, k in pairs(opt.nTestShot) do 
            local input, target = util.extractK(trainData.input, trainData.target, 
               k, opt.nClasses.test)
            
            -- minibatch loop
            local nBatches = testSet:size()  
            for idx=1,nBatches do 
               -- process test example
               local testData = testSet:get() 
               local pred = network.forward(network.params, 
                  {input, target, testData.input}) 
               for i=1,pred:size(1) do 
                  testConf[k]:add(pred[i], testData.target[i])
               end

               testConf[k]:updateValids()
               acc[k][d] = testConf[k].totalValid*100
               testConf[k]:zero()
            end   
         end

      end   

      for _,k in pairs(opt.nTestShot) do 
         print('Test Accuracy (' .. n .. ' episodes, ' .. k .. '-shot)')
         print(acc[k]:mean()) 
      end
          
      ret[n] = _.values(_.map(acc, function(i, val)
            local low = val:mean() - 1.96*(val:std()/math.sqrt(val:size(1)))
            local high = val:mean() + 1.96*(val:std()/math.sqrt(val:size(1)))
            return i .. '-shot: ' .. val:mean() .. '; ' .. val:std() 
               .. '; [' .. low .. ',' .. high .. ']'
      end)) 
   end)

   return ret
end
