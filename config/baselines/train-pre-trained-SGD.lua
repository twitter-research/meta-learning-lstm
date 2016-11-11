
return function(opt)

   opt.trainFull = true 
   opt.nClasses.train = 64
   opt.nAllClasses = 64
   
   opt.episodeSamplerKind = 'uniform'

   opt.preTrainSGD = true
   opt.model = 'model.imagenet-net'
   opt.learner = 'model.baselines.pre-trained-SGD'
   opt.nValidationEpisode = 100

   opt.learningRate = 0.01
   opt.learningRateDecay = 1e-4
   opt.batchSize = 32
   opt.nEpochs = 50000
   
   opt.learningRates = {0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001}  
   opt.learningRateDecays = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0}
   opt.nUpdates = 15 
   opt.useCUDA = false 

   opt.printPer = 1000

   return opt
end
