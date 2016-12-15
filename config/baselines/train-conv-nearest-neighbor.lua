
return function(opt)

   opt.trainFull = true 
   opt.nClasses.train = 32 --64 --20 --16
   opt.nAllClasses = 3006 --64 --4112 
  
   opt.episodeSamplerKind = 'permutation'

   --opt.normalizeData = false 
   --opt.model = 'model.conv-net'
   --opt.model = 'model.imagenet-net'
   opt.model = 'model.amazonCat-net'
   opt.learner = 'model.baselines.conv-nearest-neighbor'
   opt.nValidationEpisode = 100
   
   --opt.learningRate = 0.01 
   --opt.learningRateDecay = 1e-4
   opt.learningRate = 0.001
   opt.trainBatchSize = 32 --100 -- 32 
   opt.nEpochs = 30000    
   opt.useCUDA = true 

   opt.printPer = 200 --100 --1000 --500 --100 --500 

   return opt
end
