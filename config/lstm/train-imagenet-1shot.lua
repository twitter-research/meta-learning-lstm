
return function(opt)
   
   opt.learner = 'model.lstm-classifier'
   opt.metaLearner = 'model.lstm.train-lstm'
   opt.BN_momentum = 0.9
 
   opt.optimMethod = 'adam'
   opt.learningRate = 0.001
   opt.maxGradNorm = 0.25 
 
   opt.batchSize = {}
   opt.batchSize[1]=5
   opt.batchSize[5]=5 

   opt.nEpochs = {}
   opt.nEpochs[1]=12
   opt.nEpochs[5]=5
 
   opt.nEpisode = 75000 
   opt.nValidationEpisode = 100
   opt.printPer = 1000

   opt.useCUDA = true  

   return opt
end
