local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
local nExamples = 20

return function(opt, dataFile, episodeFile)
   local opt = opt or error("You must specify options")
   require(opt.homePath .. 'dataset.dataset') 

   -- Retrieve folder
   if not path.exists(opt.dataFolder) then
      print('Retrieving datasets...')
      local hdfs = require 'hdfs.HDFS'()
      hdfs.get(opt.datasetHdfsRoot, opt.dataFolder, true)
   end   

   -- Load raw tensor data in torch
   local rawData = util.localize(torch.load(opt.dataFolder .. '/' .. dataFile), opt)
   torch.manualSeed(123)   

   local function nDatasets()
      return #rawData
   end
   
   -- Does episode file with list of classes for each episode exist?
   local getEpisode 
   local episodeNum, episodesList
   if episodeFile then
      episodesList = torch.load(opt.dataFolder .. '/' .. episodeFile)

      episodeNum = 1
      getEpisode = 
         function()
            local list = episodesList[episodeNum]
            episodeNum = episodeNum + 1
                     
            return list
         end
   else
      getEpisode = 
         function() 
            return torch.randperm(nDatasets())
         end
   end

   -- set up data
   local data = {}
   data.trainX = util.localize(torch.Tensor(), opt)
   data.trainY = util.localize(torch.Tensor(), opt)
   data.testX = util.localize(torch.Tensor(), opt)
   data.testY = util.localize(torch.Tensor(), opt)

   local collectX, collectY   
      
   local function normalize(X)
      if #X:size() > 0 then
         local std = X:std()
         local mean = X:mean()
         X:add(-mean):mul(1/std)
      end
      return X
   end

   local function createEpisode(lopt)  
      local nClasses = lopt.nClasses
      if nClasses == nil then nClasses = nDatasets() end 
      local nTrain = lopt.nTrain or 0
      assert(nClasses <= nDatasets())
      assert(nTrain <= nExamples)
      
      local nTest = nExamples - nTrain       
      collectX = collectX or util.localize(torch.Tensor(nExamples * nClasses, 1, opt.nIn, opt.nIn), opt)
      collectY = collectY or util.localize(torch.Tensor(nExamples * nClasses), opt)

      -- fetch data from selected classes
      local selClasses = getEpisode() 
      for i=1,nClasses do
         local idx = (i-1) * nExamples + 1
         collectX[{{idx, idx+nExamples-1}}] = rawData[selClasses[i]]
         collectY[{{idx, idx+nExamples-1}}] = i
      end

      -- split into train+test 
      data.trainX:resize(nTrain*nClasses, 1, opt.nIn, opt.nIn)
      data.trainY:resize(nTrain*nClasses) 
      data.testX:resize(nTest*nClasses, 1, opt.nIn, opt.nIn)
      data.testY:resize(nTest*nClasses)
   
      for i=1,nClasses do
         local shuffle = torch.randperm(nExamples) 
         local idx=1
         for j=1,nTrain do
            data.trainX[(i-1)*nTrain + j] = collectX[(i-1)*nExamples + shuffle[idx]]
            data.trainY[(i-1)*nTrain + j] = collectY[(i-1)*nExamples + shuffle[idx]]
            idx = idx+1
         end

         for j=1,nTest do 
            data.testX[(i-1)*nTest + j] = collectX[(i-1)*nExamples + shuffle[idx]]
            data.testY[(i-1)*nTest + j] = collectY[(i-1)*nExamples + shuffle[idx]]
            idx = idx+1
         end
      end

      -- normalize data and create train+test dataset
      local trainData, testData
      trainData = Dataset({
         x = normalize(data.trainX), y = data.trainY, 
         batchSize = lopt.trainBatchSize, shuffle = true
      })
      if #data.testX:size() > 0 then 
         testData = Dataset({
            x = normalize(data.testX), y = data.testY, 
            batchSize = lopt.testBatchSize, shuffle = true
         })
      end

      return trainData, testData 
   end
   
   return { nDatasets=nDatasets, createEpisode=createEpisode } 
end
