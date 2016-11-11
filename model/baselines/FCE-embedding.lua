local t = require 'torch'
local autograd = require 'autograd'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'

return function(opt)
   local model = {}
   local maxGradNorm = opt.maxGradNorm or 0.25

   -- load and functionalize embedding nets
   local model1 = require(opt.homePath .. opt.model)({
      nClasses=opt.nClasses.train, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
   })  
   local embedNet1 = model1.net
   local model2 = require(opt.homePath .. opt.model)({
      nClasses=opt.nClasses.train, useCUDA=opt.useCUDA, classify=false, nIn=opt.nIn, nDepth=opt.nDepth
   })  
   local embedNet2 = model2.net
   local modelF, paramsF = autograd.functionalize(embedNet1)
   local modelG, paramsG = autograd.functionalize(embedNet2)   
         
   local attLSTM, paramsAttLSTM, layers = require(opt.homePath .. '.model.lstm.RecurrentLSTMNetwork')({
      inputFeatures = model1.outSize + model2.outSize,
      hiddenFeatures = model1.outSize,
      outputType = 'all'
   })

   local biLSTMForward, paramsBiLSTMForward, layers = require(opt.homePath .. '.model.lstm.RecurrentLSTMNetwork')({
      inputFeatures = model2.outSize, 
      hiddenFeatures = model2.outSize,
      outputType = 'all'
   }) 

   local biLSTMBackward, paramsBiLSTMBackward, layers = require(opt.homePath .. '.model.lstm.RecurrentLSTMNetwork')({
      inputFeatures = model2.outSize, 
      hiddenFeatures = model2.outSize,
      outputType = 'all'
   })

   local softmax = autograd.functionalize(nn.SoftMax(true)) 

   model.params = {
      f=paramsG,
      g=paramsG,
      biLSTMForward=paramsBiLSTMForward,
      biLSTMBackward=paramsBiLSTMBackward,
      attLSTM=paramsAttLSTM
   }
   
   -- set training or evaluate mode
   model.set = function(mode)
      if mode == 'training' then
         modelF.module:training()
         modelG.module:training()
      elseif mode == 'evaluate' then
         modelF.module:evaluate()
         modelG.module:evaluate()
      end
   end

   model.attLSTM = function(params, input, K)
      local gS = input[1]  
      local fX = input[2] 

      local h = {}
      local c = {}
      local r = {}
      
      --r[1] = t.zero(fX.new(t.size(fX,1), t.size(fX,2)))
      --r[1] = t.clone(fX)
      r[1] = t.expandAs(t.mean(gS,1), fX)
      for i=1,K do   
         local x = t.cat(fX, r[i], 2) 
         local tempH, state = attLSTM(params, t.view(x, t.size(x,1), 1, t.size(x,2)), {h=h[i-1], c=c[i-1]})
         h[i] = fX + t.view(tempH, t.size(tempH, 1), t.size(tempH, 3))
         c[i] = state.c
   
         local batchR = {}
         for j=1,t.size(h[i], 1) do
            local hInd = h[i][j] 
            local weight = t.sum(t.cmul(gS, t.expandAs(t.reshape(hInd, 1, hInd:size(1)), gS)), 2)
            local embed = t.sum(t.cmul(t.expandAs(softmax(weight), gS), gS), 1)
            batchR[j] = embed 
         end
         r[i+1] = t.cat(batchR, 1)

      end

      return h[#h]
   end

   model.biLSTM = function(params, input) 
      local gX = input  
   
      local fH = {}
      local stateF = {}
      local bH = {}  
      local stateB = {}    

      local gS = {}
      
      local size = t.size(gX,1)
      
      for i=1,size do
         local tempHF, tempStateF = biLSTMForward(params[1], t.index(gX, 1, t.LongTensor{i}), stateF[i-1])
         fH[i] = tempHF
         stateF[i] = tempStateF
         
         local tempHB, tempStateB = biLSTMBackward(params[2], t.index(gX, 1, t.LongTensor{size-(i-1)}), stateB[i-1]) 
         bH[i] = tempHB
         stateB[i] = tempStateB
      end

      for i=1,size do   
         gS[i] = t.index(gX, 1, t.LongTensor{i}) + fH[i] + bH[i]
      end   
      return torch.cat(gS, 1)
   end

   model.embedS = function(params, input)
      local g = modelG(params.g, input)
      return model.biLSTM({params.biLSTMForward, params.biLSTMBackward}, g)
   end

   model.embedX = function(params, input, g, K)
      local f = modelF(params.f, input)
      return model.attLSTM(params.attLSTM, {g, f}, K) 
   end
   
   model.df = function(dfDefault)
      return function(params, input, testY)
         local grads, loss, pred = dfDefault(params, input, testY)
      
         local norm = 0 
         for i,grad in ipairs(autograd.util.sortedFlatten(grads.attLSTM)) do
            norm = norm + torch.sum(torch.pow(grad,2))
         end 
         norm = math.sqrt(norm)
         if norm > maxGradNorm then
            for i,grad in ipairs(autograd.util.sortedFlatten(grads.attLSTM)) do
               grad:mul( maxGradNorm / norm )
            end 
         end 

         local norm = 0
         for i,grad in ipairs(autograd.util.sortedFlatten(grads.biLSTM)) do
            norm = norm + torch.sum(torch.pow(grad,2))
         end
         norm = math.sqrt(norm)
         if norm > maxGradNorm then
            for i,grad in ipairs(autograd.util.sortedFlatten(grads.biLSTM)) do
               grad:mul( maxGradNorm / norm )
            end
         end

         return grads, loss, pred
      end
   end
   
   model.save = function()
      local models = {modelF=embedNet1:clearState(), modelG=embedNet2:clearState()}
      torch.save('matching-net-models.th', models)
   end

   model.load = function(networkFile, opt)
      local data = torch.load(networkFile)
      modelF, _ = autograd.functionalize(util.localize(data.modelF, opt))
      modelG, _ = autograd.functionalize(util.localize(data.modelG, opt)) 
   end


   return model
end
