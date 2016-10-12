
return function(opt)

   opt.trainFull = true 
   opt.nClasses.train = 16
   opt.nAllClasses = 4112 
   --opt.episodeSamplerKind = 'uniform'

   opt.preTrainSGD = true
   opt.model = 'model.med-conv-net'
   opt.learner = 'model.baselines.pre-trained-SGD'
   opt.nValidationEpisode = 100

   opt.learningRate = 0.01
   opt.learningRateDecay = 1e-4
   opt.batchSize = 32
   opt.nEpochs = 30000
   
   opt.learningRates = {0.1, 0.01, 0.001, 0.0001, 0.00001}  
   opt.learningRateDecays = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5}
   opt.nUpdates = {1,2,3,4,5}
   opt.useCUDA = true 

   opt.printPer = 100

   return opt
end
