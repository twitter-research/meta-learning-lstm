require 'cortex-core.projects.research.oneShotLSTM.model.lstm.meta-learner-lstm'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'
require 'gnuplot'
local autograd = require 'autograd'

_ = require 'moses'

-- parse command line arguments
cmd = torch.CmdLine()
cmd:option('--params', '', 'meta-learner params')
cmd:option('--model', '', 'learner net')
cmd:option('--data', '', 'data')
cmd:option('--task', '', 'task')
cmd:option('--nEpisodes', 1, 'number of episodes')

local argVals = cmd:parse(arg)
local opt = {}
opt = require('cortex-core.projects.research.oneShotLSTM.' .. argVals.task)(opt)
opt = require('cortex-core.projects.research.oneShotLSTM.' .. argVals.data)(opt)
opt = require('cortex-core.projects.research.oneShotLSTM.' .. argVals.model)(opt)
local metaLearnerParams = torch.load(argVals.params)

local learner = getLearner(opt)
local metaLearner = getMetaLearner2({learnerParams=learner.params, nParams=learner.nParams, debug=opt.debug, homePath=opt.homePath, BN1=opt.BN1, BN2=opt.BN2})

print('Meta Learner params: ')
print(metaLearnerParams)

local M = torch.zeros(argVals.nEpisodes, opt.nEpochs[opt.nTrainShot] * (opt.nClasses.train * opt.nTrainShot)/opt.batchSize[opt.nTrainShot]) 

local f_data = {M:clone(), M:clone(), M:clone(), M:clone(), M:clone()}
local i_data = {M:clone(), M:clone(), M:clone(), M:clone(), M:clone()}

-- load data
local train, validation, test = require(opt.homePath .. opt.dataLoader)(opt)


local layers = {1,5,9,13,17}
local rand_int1 = {}
local rand_int2 = {}
for i=1,#layers do
   if layers[i] == 1 then
      rand_int1[layers[i]] = torch.random(1,32)
      rand_int2[layers[i]] = torch.random(1,3)
   elseif layers[i]==5 or layers[i]==9 or layers[i]==13 then
      rand_int1[layers[i]] = torch.random(1,32)
      rand_int2[layers[i]] = torch.random(1,32)
   else
      rand_int1[layers[i]] = torch.random(1,5)
      rand_int2[layers[i]] = torch.random(1,800)
   end
end


for n=1,argVals.nEpisodes do  
   print('Episode: ' .. n)
   local trainSet, testSet = validation.createEpisode({})
   local trainData = trainSet:get()
   local testData = testSet:get()
    
   local trainInput, trainTarget = util.extractK(trainData.input, trainData.target, opt.nTrainShot, opt.nClasses.val)   

   local learnerParams = torch.clone(metaLearnerParams[2].cI)  
   local metaLearnerState = {}
   local metaLearnerCell = {}

   local batchSize = opt.batchSize[opt.nTrainShot]

   local trainSize = trainInput:size(1)    
   local idx = 1 
   for s=1,opt.nEpochs[opt.nTrainShot] do
      for i=1,trainSize,batchSize do  
         -- get image input & label
         local x = trainInput[{{i,i+batchSize-1},{},{},{}}]
         local y = trainTarget[{{i,i+batchSize-1}}]

         -- get gradient and loss w/r/t input+label      
         local gradLearner, lossLearner = learner.df(learnerParams, x, y)    

         -- preprocess grad & loss 
         gradLearner = torch.view(gradLearner, gradLearner:size(1), 1, 1)
         local preGrad, preLoss = preprocess(gradLearner, lossLearner)

         -- use meta-learner to get learner's next parameters
         local state = metaLearnerState[idx-1] or {{},{}} 
         local cOut, sOut = metaLearner.forward(metaLearnerParams, layers, {preLoss, preGrad, gradLearner}, state)   
         metaLearnerState[idx] = sOut 
         metaLearnerCell[idx] = cOut

         -- break computational graph with getValue call 
         learnerParams = cOut
         idx = idx + 1 
      end 
   end
 
   for i=1,#metaLearnerState do
      local learnerParamsFormatF = learner.unflattenParams(metaLearnerState[i][2].f)
      local learnerParamsFormatI = learner.unflattenParams(metaLearnerState[i][2].i)
      print(learnerParamsFormatF)

      for j=1,#layers do 
         local fval, ival
         if layers[j] ~= 17 then
            fval = learnerParamsFormatF[layers[j]][rand_int1[layers[j]]][rand_int2[layers[j]]][1][1]
            ival = learnerParamsFormatI[layers[j]][rand_int1[layers[j]]][rand_int2[layers[j]]][1][1]    
            --table.insert(f_data[j], learnerParamsFormatF[layers[j]][1][1][1][1]
            --table.insert(i_data[j], learnerParamsFormatI[layers[j]][1][1][1][1]
         else
            fval = learnerParamsFormatF[layers[j]][rand_int1[layers[j]]][rand_int2[layers[j]]]
            ival = learnerParamsFormatI[layers[j]][rand_int1[layers[j]]][rand_int2[layers[j]]]
            --table.insert(f_data[j], learnerParamsFormatF[layers[j]][1][1])
            --table.insert(i_data[j], learnerParamsFormatI[layers[j]][1][1])
         end
         --print(f_data[j])
         --print(n)
         --print(i)
         f_data[j][{{n},{i}}] = fval
         i_data[j][{{n},{i}}] = ival
      end    
   end
end

function mean(listM)
   local l = {}
   for i=1,#listM do
      l[i] =  torch.mean(autograd.util.sigmoid(listM[i]), 1):resize(listM[i]:size(2))
   end

   return l
end

function std(listM)
   local l = {}
   for i=1,#listM do
      l[i] =  torch.std(autograd.util.sigmoid(listM[i]), 1):resize(listM[i]:size(2))
   end

   return l
end

function getSig(listM, idx)
   local l = {}
   for i=1,#listM do
      l[i] = autograd.util.sigmoid(listM[i][idx]) 
   end

   return l
end

local perm = torch.randperm(argVals.nEpisodes)
local dataF = {mean(f_data), std(f_data)}
local dataI = {mean(i_data), std(i_data)}
for i=1,20 do
   table.insert(dataF, getSig(f_data, perm[i]))
   table.insert(dataI, getSig(i_data, perm[i]))
end

torch.save('results/' .. argVals.task .. '_' .. argVals.data .. '_' .. argVals.model .. '_forgetGate.t7', 
   dataF) 
torch.save('results/' .. argVals.task .. '_' .. argVals.data .. '_' .. argVals.model .. '_inputGate.t7', 
   dataI)

--[[torch.save('results/' .. argVals.task .. '_' .. argVals.data .. '_' .. argVals.model .. '_forgetGate.t7' , f_data)
torch.save('results/' .. argVals.task .. '_' .. argVals.data .. '_' .. argVals.model .. '_inputGate.t7' , i_data)
--]]
--[[local data = {}
for i=1,#f_data do
   table.insert(data, {'layer: ' .. layers[i], autograd.util.sigmoid(torch.FloatTensor(f_data[i]))}) 
end

gnuplot.pngfigure('graph.png')
gnuplot.plot(data)
gnuplot.plotflush()
--]]
