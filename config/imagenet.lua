
return function(opt)
   opt.nExamples = 20
   opt.nDepth = 3
   opt.nIn = 84   

   opt.dataName = 'dataset.miniImagenet'
   opt.dataLoader = 'dataset.data-loader2'   

   opt.rawDataDir = 'raw-data/'

   return opt
end
