
return function(opt)

	opt.trainFull = true	
	opt.nClasses.train = 20 --16
	opt.nAllClasses = 64 --4112 
	opt.episodeSamplerKind = 'uniform'

	opt.model = 'model.conv-net'
	opt.learner = 'model.baselines.conv-nearest-neighbor'
	opt.nValidationEpisode = 100
	
	opt.learningRate = 0.01	
	opt.learningRateDecay = 1e-4
	opt.trainBatchSize = 100 -- 32 
	opt.nEpochs = 100000		
	opt.useCUDA = true 

	opt.printPer = 100 --500 

	return opt
end
