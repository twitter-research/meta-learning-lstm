
return function(opt)
   opt.nExamples = 20
   opt.nDepth = 1
   opt.nIn = 2000   

   opt.dataName = 'dataset.amazonCat'
   opt.dataLoader = 'dataset.data-loader2'   

   opt.rawDataDir = 'raw-data/'
   opt.meanStdNormalize = true

   return opt
end
