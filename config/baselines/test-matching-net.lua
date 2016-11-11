
return function(opt)
   opt.nEpisode = 0
   --opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/5shot_omniglot_metaLearner_params.th'
   --opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/mini-Imagenet/matching-net_params_snapshot.th'
   --opt.paramsFile = 'saved_params/random_shot_25_imagenet_metaLearner_params.th' 

   --opt.paramsFile = 'saved_params/mini-Imagenet/matching-net-simple/matching-net_params_snapshot.th'
   --opt.networkFile = 'saved_params/mini-Imagenet/matching-net-simple/matching-net-models.th'

   opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/mini-Imagenet/matching-net-simple/matching-net_params_snapshot.th' 
   opt.networkFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/mini-Imagenet/matching-net-simple/matching-net-models.th'

   return opt
end
