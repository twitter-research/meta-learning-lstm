
return function(opt)
   
   opt.learner = 'model.lstm-classifier'
   opt.metaLearner = 'model.lstm.train-lstm'
   opt.BN_momentum = 0.95
    
   opt.optimMethod = 'adam'
   opt.learningRate = 0.001
   opt.maxGradNorm = 0.25 
 
   opt.batchSize = {}
   opt.batchSize[1]=5
   opt.batchSize[5]=25  

   opt.nEpochs = {}
   opt.nEpochs[1]=5
   opt.nEpochs[5]=8 
 
   opt.nEpisode = 50000 
   opt.nValidationEpisode = 100
   opt.printPer = 1000 

   opt.useCUDA = true  

   return opt
end
