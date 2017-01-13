
return function(opt)
  
   opt.learner = 'model.matching-net-classifier'
   opt.metaLearner = 'model.baselines.pixel-nearest-neighbor'
 
   opt.trainFull = true 
   opt.nClasses.train = 64 --20 --16
   opt.nAllClasses = 64  --4112 
    
   opt.useCUDA = false

   return opt
end
