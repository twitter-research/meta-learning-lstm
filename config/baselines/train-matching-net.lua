
return function(opt) 

   opt.dataLoader = 'dataset.data-loader2'

   opt.model = 'model.conv-net'
   opt.learner = 'model.baselines.matching-net'
   opt.embedModel = 'model.baselines.simple-embedding'
   --opt.embedModel = 'model.baselines.FCE-embedding'

   opt.episodeSamplerKind = 'uniform'

   opt.optimMethod = 'adam'   
   opt.learningRate = 0.0001

   --[[
   opt.optimMethod = 'sgd'
   opt.learningRate = 0.1
   opt.momentum = 0.9
   opt.nesterov = true
   opt.dampening = 0
   --]]

   --opt.learningRateDecay = 1e-5
   --opt.momentum = 0.9
   --opt.weightDecay = 0.0005 

   opt.nEpisode = 75000 --4800  --150000 
   opt.nValidationEpisode = 100 
   
   opt.batchSize = opt.nClasses.train * opt.nEval 
   --opt.batchSize = 2
   opt.episodeBatchSize = 1 

   opt.nEpochs = 1 --25 --1
   opt.printPer = 1000

   opt.useCUDA = true 

   return opt
end
