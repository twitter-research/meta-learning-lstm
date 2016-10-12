
return function(opt)
	
	opt.trainFull = true 
   opt.nClasses.train = 16
   opt.nAllClasses = 4112 
   --opt.episodeSamplerKind = 'uniform'
	
	opt.model = 'model.conv-net'
	opt.learner = 'model.baselines.pixel-nearest-neighbor'		
	opt.useCUDA = false

	return opt
end
