
return function(opt)
   opt.nClasses = {train=50, val=50, test=50}
   opt.nTrainShot = 1
   opt.nEval = 15
   
   opt.nTest = {100, 250, 600}
   opt.nTestShot = {1,5}

   return opt
end
