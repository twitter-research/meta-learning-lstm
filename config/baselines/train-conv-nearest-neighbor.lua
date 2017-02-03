
return function(opt)

   opt.learner = 'model.matching-net-classifier'
   opt.metaLearner = 'model.baselines.conv-nearest-neighbor'

   opt.trainFull = true 
   opt.nClasses.train = 64 
   opt.nAllClasses = 64   
    
   opt.learningRate = 0.001
   opt.trainBatchSize = 32 
   
   opt.nEpochs = 30000    
   opt.nValidationEpisode = 100
   opt.printPer = 1000 
   
   opt.useCUDA = true 

   return opt
end
