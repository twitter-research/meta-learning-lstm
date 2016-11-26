local nn = require 'nn'
local optim = require 'optim'
local util = require 'util.util'
_ = require 'moses'

return function(opt)  
   -- load config info for task, data, and model 
   opt = require(opt.task)(opt)
   opt = require(opt.data)(opt)
   opt = require(opt.model)(opt)
   if opt.test ~= '-' then
      opt = require(opt.test)(opt)
   end

   -- options 
   print('Training with options:', _.sort(opt))

   -- load cunn
   if opt.useCUDA then 
      require 'cunn' 
      require 'cutorch'
   end

   -- set up train, validation & test datasets  
   print('Using data-loader: ' .. opt.dataLoader)
   local train, validation, test
   if opt.dataLoader == 'dataset.data-loader' then 
      train = require(opt.dataLoader)(opt, opt.trainFile)  
      validation = require(opt.dataLoader)(opt, opt.validationFile, opt.validationEpisodesFile)
      test = require(opt.dataLoader)(opt, opt.testFile, opt.testEpisodesFile)
   elseif opt.dataLoader == 'dataset.data-loader2' then
      train, validation, test = require(opt.dataLoader)(opt)
   end
   local data = {train=train, validation=validation, test=test}   
   
   -- 
   for k,v in pairs(opt) do
      if v=="true" or v=="false" then
         opt[k] = util.boolize(opt[k])
      end
   end
      
   -- run
   local acc = require(opt.learner)(opt, data)
   return acc

end

--[[ WHETLAB
   opt.done = nil
   if opt.useWHETLAB then
      local scientist, job, done = require 'cortex-core.projects.research.oneShotLSTM.config.scientist' ({
         parameters = {
            optimMethod = {type='E', size=1, options={'adam', 'sgd'}},
            learningRate = {type='F', min=1e-6, max=1e-1, size=1}, 
            lrDecay = {type='F', min=1e-6, max=1e-2, size=1},
            maxGradNorm = {type='F', min=0.10, max=5.0, size=1},
            BN = {type='E', size=1, options={'false', 'true'}},
            nEpochs = {type='I', min=1, max=3, size=1}   
         },
         objective = {min=0.0, max=1.0, target=0.8},
         resume = true,
         constraints = { isValid={type='B', size=1} },
         name = 'LSTM Hyperparameter selection',
         tags = {'LSTM'},
         description = [[
            Command line:
            torch --cluster atla --gpus 1 --dedicated cortex-k80 --job LSTM-whetlab --instances 7 --service -- cortex-core.projects.research.oneShotLSTM.train.run-train --task config.5-shot --data config.omniglot --model config.lstm.train-lstm
         ,  
      })

      opt.optimMethod = job.optimMethod[1]
      opt.learningRate = job.learningRate[1]  
      opt.lrDecay = job.lrDecay[1]   
      opt.maxGradNorm = job.maxGradNorm[1]
      opt.BN = job.BN[1]
      opt.nEpochs = job.nEpochs[2]

      opt.done = done
   end
--]]
