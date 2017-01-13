local t = require 'torch'
local autograd = require 'autograd'
local funcNN = autograd.functionalize('nn') 
local util = require 'util.util'

local P = 10
local expP = torch.exp(P)
local negExpP = torch.exp(-P)

function preProc1(x)
   local absX = t.abs(x) 
   local cond1 = torch.gt(absX, negExpP)
   local cond2 = torch.le(absX, negExpP)

   local x1 = x:maskedSelect(cond1)
   x1 = t.log(t.abs(x1))/P
   local x2 = x:maskedSelect(cond2)
   x2:fill(-1) 

   local z = util.zerosAs(x, x:size())
   z:maskedCopy(cond1, x1) 
   z:maskedCopy(cond2, x2) 

   return z 
end

function preProc2(x)
   local absX = t.abs(x)
   local cond1 = torch.gt(absX, negExpP)
   local cond2 = torch.le(absX, negExpP)

   local x1 = x:maskedSelect(cond1)
   x1 = t.sign(x1) 
   local x2 = x:maskedSelect(cond2)
   x2 = x2*expP

   local z = util.zerosAs(x, x:size())
   z:maskedCopy(cond1, x1) 
   z:maskedCopy(cond2, x2) 

   return z 
end

-- pre-processing according to Deepmind 'Learning to Learn' paper
function preprocess(grad, loss)
   local preGrad = torch.zero(grad.new(grad:size(1), 1, 2))
   preGrad[{{},{},{1}}] = preProc1(grad)
   preGrad[{{},{},{2}}] = preProc2(grad)  

   local lossT = util.zerosAs(grad, torch.LongStorage({1,1,1}))
   lossT[1] = loss   
   local preLoss = util.zerosAs(grad, torch.LongStorage({1,1,2}))
   preLoss[{{},{},{1}}] = preProc1(lossT) 
   preLoss[{{},{},{2}}] = preProc2(lossT)

   return preGrad, preLoss 
end

---------------------------------------------------
-- meta-learner LSTM
function getMetaLearnerLSTM(opt, params, layers)   
   
   opt = opt or {}
   local nParams = opt.nParams 
   local m = 0 
   local nInput = opt.nInput  

   local params = params or {}
   local layers = layers or {}
   local l = {}

   -- params
   local p = {
      WF = torch.zeros(nInput+2,1),
      WI = torch.zeros(nInput+2,1),
      cI = torch.zeros(nParams,1),  -- initial cell state is a param 
      bI = torch.zeros(1,1),
      bF = torch.zeros(1,1)
   }

   table.insert(params, p)

   -- function:
   -- x is table of {loss, gradient}
   local f = function(params, input, prevState, layers)  
      local x_all = input[1]
      local grad_input = input[2]
   
      local batch = t.size(grad_input, 1)
      local steps = t.size(grad_input, 2) 

      -- hiddens
      prevState = prevState or {}
      local fS = {}
      local iS = {}
      local cS = {}
      local deltaS = {}

      -- go over time steps of input
      for s=1,steps do
         -- loss, gradient inputs
         local x = t.select(x_all, 2, s)
         local act_g = t.select(grad_input, 2, s)  
       
         -- prev f, i, and c value
         local fP = fS[s-1] or prevState.f or torch.zero(grad_input.new(batch,1)) 
         local iP = iS[s-1] or prevState.i or torch.zero(grad_input.new(batch,1))
         local cP = cS[s-1] or prevState.c or params.cI  
         local deltaP = deltaS[s-1] or prevState.delta or 
            torch.zero(grad_input.new(batch,1)) 

         -- next forget, input gate
         local fH = t.cat(cP, fP ,2)
         local iH = t.cat(cP, iP, 2)
         
         local fN = t.cat(x, fH, 2) * params.WF + t.expand(params.bF, batch, 1)
         local iN = t.cat(x, iH ,2) * params.WI + t.expand(params.bI, batch, 1)
         fS[s] = fN  
         iS[s] = iN
      
         -- next delta
         local delta = 
            m * deltaP - t.cmul(autograd.util.sigmoid(t.view(iN,nParams,1)), act_g)

         -- next cell/params
         cS[s] = t.cmul(autograd.util.sigmoid(t.view(fN,nParams,1)), cP) + delta 
         
      end

      -- save state
      local newState = {f=fS[#fS], i=iS[#iS], c=cS[#cS]}
      -- return last cell
      return cS[#cS], newState   
   end

   return f, params, layers
end
