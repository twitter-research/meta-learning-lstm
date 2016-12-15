
return function(opt)

   opt.trainFull = true 
   opt.nClasses.train = 64
   opt.nAllClasses = 64
   
   opt.episodeSamplerKind = 'permutation'

   opt.preTrainSGD = true
   opt.model = 'model.conv-net'
   --opt.model = 'model.imagenet-net'
   opt.learner = 'model.baselines.pre-trained-SGD'
   opt.nValidationEpisode = 100
 
   opt.learningRates = {0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001}  
   opt.learningRateDecays = {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0}
   --opt.learningRates = {0.1}
   --opt.learningRateDecays = {0}
   
   opt.nUpdates = {15} 
   opt.useCUDA = true 

   opt.printPer = 1000

   return opt
end
