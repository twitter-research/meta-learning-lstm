
return function(opt)
   opt.nEpochs = 0
   --opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/5shot_omniglot_metaLearner_params.th'
   opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/omniglot/conv-nearest-neighbor-model.th'
   --opt.paramsFile = 'saved_params/random_shot_25_imagenet_metaLearner_params.th' 

   return opt
end
