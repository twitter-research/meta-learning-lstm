
return function(opt)

   opt.learner = 'model.matching-net-classifier'
   opt.metaLearner = 'model.baselines.conv-nearest-neighbor'

   opt.trainFull = true 
   opt.nClasses.train = 64 
   opt.nAllClasses = 64  
  
   --opt.normalizeData = false 
   --opt.model = 'model.conv-net'
   --opt.model = 'model.imagenet-net'
   
   --opt.learningRate = 0.01 
   --opt.learningRateDecay = 1e-4
   opt.learningRate = 0.001
   opt.trainBatchSize = 32 --100 -- 32 
   
   opt.nEpochs = 75000    
   opt.nValidationEpisode = 100
   opt.printPer = 1000 --100 --1000 --500 --100 --500 
   
   opt.useCUDA = true 

   return opt
end
