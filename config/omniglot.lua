
return function(opt)
   opt.nExamples = 20
   opt.nDepth = 1
   opt.nIn = 28   

   opt.dataName = 'omniglot'
   opt.homePath = 'cortex-core.projects.research.oneShotLSTM.'
   opt.dataLoader = 'dataset.data-loader2'
   
   return opt
end
