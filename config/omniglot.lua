
return function(opt)
   opt.nExamples = 20
   opt.nDepth = 1
   opt.nIn = 28   

   opt.dataName = 'omniglot'
   opt.homePath = 'cortex-core.projects.research.oneShotLSTM.'
   opt.dataLoader = 'dataset.data-loader2'
  
   --[[
   opt.dataLoader = 'dataset.data-loader'
   opt.datasetHdfsRoot = '/user/cortex/cxbenchmarks/images/omniglot/'
   opt.dataFolder = 'omniglot'
   opt.trainFile = 'train.th'
   opt.validationFile = 'validation.th'
   opt.testFile = 'test.th'
   --]]

   return opt
end
