
return function(opt)
   opt.nEpisode = 0 

   opt.paramsFile = 'saved_params/miniImagenet/matching-net-FCE/matching-net_params_snapshot.th'
   opt.networkFile = 'saved_params/miniImagenet/matching-net-FCE/matching-net-models.th'

   return opt
end
