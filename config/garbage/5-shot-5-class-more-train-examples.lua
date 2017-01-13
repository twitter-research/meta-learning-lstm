
return function(opt)
   opt.nClasses = {train=5, val=5, test=5}
   opt.nTrainShot = 5
   opt.nEval = 15

   opt.nTest = {500}
   --opt.nTestShot = {1,5,25,100,250,500}
   --opt.nTestShot = {1,5,10,25,50}
   --opt.nTestShot = {1, 5, 25, 100}
   opt.nTestShot = {100, 250}

   return opt
end
