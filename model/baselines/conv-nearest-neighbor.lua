
return function(opt, dataset)
   opt.convNearestNeighbor = true
   opt.embedFunction = function(lopt)
      local network = lopt.network
      local trainData = lopt.trainData
      local testData = lopt.testData

      -- embeddings are output of pre-trained network
      network:evaluate()
      network:forward(trainData)
      local trainEmbedding = network.modules[#network.modules - 1].output:clone()   
      network:forward(testData)
      local testEmbedding = network.modules[#network.modules - 1].output:clone()

      return trainEmbedding, testEmbedding
   end 

   return require(opt.homePath .. 'model.baselines.pre-train')(opt, dataset)
end
