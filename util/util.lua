
local util = {}

function util.localize(x, opt)
   if x then
      if type(x) == 'table' and type(x[1]) == 'userdata' then
         for i=1,#x do 
            if opt.useCUDA then
               x[i] = x[i]:cuda()
            else
               x[i] = x[i]:float()
            end
         end
      else 
         if opt.useCUDA then
            x = x:cuda()
         else
            x = x:float()
         end
      end
   end

   return x
end

function util.paramNorm(params)
   local norms = {}
   if #params > 0 and type(params[1]) == 'table' then 
      for i, weights in pairs(params) do
         norms[i] = {}
         for j, weight in pairs(weights) do  
            norms[i][j] = math.sqrt(torch.sum(torch.pow(weight,2))) 
         end
      end
   elseif #params > 0 and type(params[1]) == 'userdata' then 
      for j, weight in pairs(params) do   
         norms[j] = math.sqrt(torch.sum(torch.pow(weight,2))) 
      end
   end
   return norms
end

function util.gradParamRatio(lr, grads, params)
   local ratios = {}
   for i, weights in pairs(params) do
      ratios[i] = {}
      for j, weight in pairs(weights) do     
         ratios[i][j] = torch.sum(torch.pow(-grads[i][j]*lr, 2))/torch.sum(torch.pow(weight, 2)) 
      end
   end 
   return ratios
end

function util.boolize(x)
    if x == 'false' then
        x = false
    else
        x = true
    end
    return x
end

function util.NaNCheck(params)
   local NaNParams = {}
   if #params > 0 and type(params[1]) == 'table' then 
      for i, weights in pairs(params) do
         for j, weight in pairs(weights) do
            local nan_mask = weight:ne(weight)
            if torch.sum(nan_mask) > 0 then
               table.insert(NaNParams, j)
            end
         end
      end
   elseif #params > 0 and type(params[1]) == 'userdata' then 
      for j, weight in pairs(params) do
         local nan_mask = weight:ne(weight)
         if torch.sum(nan_mask) > 0 then
            table.insert(NaNParams, j)
         end
      end
   end
   
   return NaNParams
end

function util.deepClone(tbl)
   if type(tbl) == "table" then
      local copy = { }
      for k, v in pairs(tbl) do
         if type(v) == "table" then
            copy[k] = util.deepClone(v)
         elseif type(v) == "number" then
            copy[k] = v
         else
            if copy[k] then
               copy[k]:copy(v)
            else
               copy[k] = v:clone()
            end
         end
      end
      return copy
   else
      return tbl
   end
end

function util.debugPrint(s, opt)
   if opt.debug then
      print(s)
   end
end

function util.zerosAs(tensor, size)
   local x
   if tensor:type() == 'torch.CudaTensor' then
      x = torch.CudaTensor(size):zero()
   elseif tensor:type() == 'torch.FloatTensor' then
      x = torch.FloatTensor(size):zero()  
   end

   return x
end

function util.getCurrentLR(opt)
   local LR 
   if opt.optimMethod == 'sgd' then
      LR = opt.currentLearningRate
   elseif opt.optimMethod == 'adam' then
      LR = opt.learningRate
   end

   return LR 
end

function util.unflattenParamsArb(exampleParams, flatParams)
   local unflattenParams = {}

   local startIdx = 1 
   for i=1,#exampleParams do
      local x = torch.view(torch.narrow(flatParams, 1, startIdx, exampleParams[i]:nElement()), exampleParams[i]:size())
      startIdx = startIdx + exampleParams[i]:nElement()
      unflattenParams[i] = x 
   end 

   return unflattenParams
end

-- extract k items from each class 
function util.extractK(input, target, k, nClasses)
   if k*nClasses == target:size(1) then
      return input, target
   end
   
   local inputNew = torch.Tensor(k*nClasses, input:size(2), input:size(3), input:size(4)):typeAs(input)
   local targetNew = torch.Tensor(k*nClasses):typeAs(target)

   local count = {}
   for i=1,nClasses do
      count[i] = 0
   end

   local idx = 1
   for i=1,target:size(1) do
      if count[target[i]] < k then
         inputNew[idx] = input[i]
         targetNew[idx] = target[i] 
         
         idx = idx + 1
         count[target[i]] = count[target[i]] + 1
      end
   end

   return inputNew, targetNew:contiguous()
end

return util
