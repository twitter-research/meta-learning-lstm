
return function(opt)
   
   opt.nExamples = 20
   opt.nDepth = 3
   opt.nIn = 84   
   
   opt.rawDataDir = 'data/miniImagenet/'   

   opt.dataName = 'dataset.miniImagenet'
   opt.dataLoader = 'dataset.data-loader'   
   opt.episodeSamplerKind = 'permutation'

   return opt
end
