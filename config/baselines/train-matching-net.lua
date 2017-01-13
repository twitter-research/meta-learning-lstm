
return function(opt) 

   opt.learner = 'model.matching-net-classifier'
   opt.metaLearner = 'model.baselines.matching-net'
   opt.embedModel = 'model.baselines.simple-embedding'
   --opt.embedModel = 'model.baselines.FCE-embedding'

   opt.optimMethod = 'adam'   
   opt.learningRate = 0.0001
   opt.batchSize = opt.nClasses.train * opt.nEval 
   
   opt.nEpisode = 75000 
   opt.nValidationEpisode = 100 
   opt.printPer = 1000 

   opt.useCUDA = true 

   return opt
end
