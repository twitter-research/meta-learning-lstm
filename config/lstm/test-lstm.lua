
return function(opt)
   opt.nEpisode = 0
   --opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/5shot_omniglot_metaLearner_params.th'
   --opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/random_shot_25_imagenet_metaLearner_params.th'
   --opt.paramsFile = 'saved_params/random_shot_25_imagenet_metaLearner_params.th' 
   
   opt.paramsFile = 'lua2/cortex-core/projects/research/oneShotLSTM/saved_params/mini-Imagenet/meta-learner-5shot/metaLearner_params_snapshot.th' 

   return opt
end
