local nearestNeighborLib = {}

function nearestNeighborLib.classifySingleTest(trainX, trainY, singleTestX, opt, K) 
   K = K or 1
   local repeatedTestX = singleTestX:repeatTensor(trainX:size(1), 1):clone()

   -- calulate cosine weights
   local m = nn.Sequential()
   m:add(nn.CosineDistance())
   
   if opt.useCUDA then 
      m = m:cuda()
   else
      m = m:float()
   end
   local w = m:forward({trainX:clone(), repeatedTestX})
   
   -- return label of most simlar neighbor 
   local _, idxs = torch.sort(-w, 1)
   return trainY[idxs[1]]
 
end

function nearestNeighborLib.classify(network, trainInput, trainLabel, testInput, opt)
   local trainEmbedding, testEmbedding = opt.embedFunction({network=network, trainData=trainInput, testData=testInput})  
   
   local ret = {}
   for i=1,testEmbedding:size(1) do 
      ret[i] = nearestNeighborLib.classifySingleTest(trainEmbedding, trainLabel, testEmbedding[i], opt)
   end

   return torch.Tensor(ret):typeAs(trainLabel)
end

return nearestNeighborLib
