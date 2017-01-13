
return function(opt)
   local nClasses = 20
   opt.nClasses = {train=nClasses, val=nClasses, test=nClasses}
   opt.nTrainShot = 5
   opt.nEval = 15

   opt.nTest = {600}
   opt.nTestShot = {1,5}

   return opt
end
