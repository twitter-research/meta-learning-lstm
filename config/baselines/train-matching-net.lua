
return function(opt) 

   opt.dataLoader = 'dataset.data-loader2'

   opt.model = 'model.conv-net'
   opt.learner = 'model.baselines.matching-net'
   opt.embedModel = 'model.baselines.simple-embedding'

   opt.optimMethod = 'adam'   
   opt.learningRate = 0.0001     

   --opt.learningRateDecay = 1e-5
   --opt.momentum = 0.9
   opt.weightDecay = 0.0005 

   opt.nEpisode = 75000 
   opt.nValidationEpisode = 200 
   opt.batchSize = opt.nClasses.train * opt.nEval 
   opt.nEpochs = 1
   opt.printPer = 200

   opt.useCUDA = false 

   return opt
end
