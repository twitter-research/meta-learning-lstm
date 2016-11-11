
return function(opt)

   opt.trainFull = true 
   opt.nClasses.train = 64 --20 --16
   opt.nAllClasses = 4112 
  
   opt.episodeSamplerKind = 'uniform'

   --opt.normalizeData = false 

   opt.model = 'model.conv-net'
   opt.learner = 'model.baselines.conv-nearest-neighbor'
   opt.nValidationEpisode = 100
   
   --opt.learningRate = 0.01 
   --opt.learningRateDecay = 1e-4
   opt.learningRate = 0.001
   opt.trainBatchSize = 32 --100 -- 32 
   opt.nEpochs = 75000    
   opt.useCUDA = true 

   opt.printPer = 1000 --500 --100 --500 

   return opt
end
