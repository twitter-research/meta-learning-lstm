
return function(opt)

   opt.learner = 'model.matching-net-classifier'
   opt.metaLearner = 'model.baselines.pre-trained-SGD'

   opt.trainFull = true 
   opt.nClasses.train = 64
   opt.nAllClasses = 64
   
   opt.learningRate = 0.001
   opt.trainBatchSize = 32
  
   opt.learningRates = {0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001}  
   opt.learningRateDecays = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0}
   --opt.learningRates = {0.1}
   --opt.learningRateDecays = {0}
   opt.nUpdates = {15} 
   
   opt.nEpochs = 50000
   opt.nValidationEpisode = 100
   opt.printPer = 1000
   
   opt.useCUDA = true 

   return opt
end
