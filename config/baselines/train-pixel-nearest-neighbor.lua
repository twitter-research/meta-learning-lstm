
return function(opt)
   
   opt.trainFull = true 
   opt.nClasses.train = 20 --16
   opt.nAllClasses = 64  --4112 
   --opt.episodeSamplerKind = 'uniform'
   opt.normalizeData = false
   
   opt.model = 'model.conv-net'
   opt.learner = 'model.baselines.pixel-nearest-neighbor'      
   opt.useCUDA = false

   return opt
end
