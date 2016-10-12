
return function(opt) 
   opt.nClasses = {train=20, val=20, test=20}
   opt.nTrainShot = 5
   opt.nEval = 15
   
   opt.nTest = 100
   opt.nTestShot = {1,5}

   return opt
end
