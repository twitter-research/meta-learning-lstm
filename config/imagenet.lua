
return function(opt)
	opt.nExamples = 20
	opt.nDepth = 3
	opt.nIn = 84   

	opt.dataName = 'miniImagenet'
	opt.homePath = 'cortex-core.projects.research.oneShotLSTM.'
	opt.dataLoader = 'dataset.data-loader2'	

	return opt
end
