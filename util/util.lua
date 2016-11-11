
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

function util.checkAndExpandParams(networkParams, loadParams)
   if type(networkParams) == 'userdata' and type(loadParams) == 'userdata' then
      
      if networkParams:nElement() ~= loadParams:nElement() then
         local oldSize = loadParams:size(1)
         local newSize = networkParams:size(1)
         
         loadParams:resizeAs(networkParams:typeAs(loadParams))
         for i=oldSize+1,newSize do
            loadParams[i][1] = torch.uniform(-0.1, 0.1)
         end
      end
   else
      error("inputs are not both tables or both tensors") 
   end
   
   --[[if type(networkParams) == 'table' and type(loadParams) == 'table' then
      print('here')
      assert(#networkParams == #loadParams, "Param table sizes should be equal")
      for i=1,#networkParams do 
         util.checkAndExpandParams(networkParams[i], loadParams[i])
      end
   elseif type(networkParams) == 'userdata' and type(loadParams) == 'userdata' then
      if networkParams:nElement() ~= loadParams:nElement() then
         loadParams:resizeAs(networkParams:typeAs(loadParams))
      end
   else
      error("inputs are not both tables or both tensors")
   end--]]

   return loadParams
end

function util.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
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
      if count[target[i]] and count[target[i]] < k then
         inputNew[idx] = input[i]
         targetNew[idx] = target[i] 
   
         idx = idx + 1
         count[target[i]] = count[target[i]] + 1
      end
   end 

   return inputNew, targetNew
end

function util.extracKSpecial(input, target, k, nClasses)
   local inputNew = torch.Tensor(k*nClasses, input:size(2), input:size(3), input:size(4)):typeAs(input)
   local targetNew = torch.Tensor(k*nClasses):typeAs(target)  
 
   local found = {}
   local idx = 1
   for i=1,target:size(1) do
      if not found[target[i]] then
         inputNew[idx] = input[i]
         targetNew[idx] = target[i]

         idx = idx + 1
         found[target[i]] = true 
      end
   end

   return inputNew, targetNew:contiguous()
end

function util.getRandomSubset(input, target, num)
   local shuffle = torch.randperm(input:size(1))
   input = input:index(1, shuffle:long())
   target = target:index(1, shuffle:long())

   return input[{{1,num},{},{},{}}], target[{{1,num}}]
end

function util.shuffleSplit(trainInput, trainTarget, testInput, testTarget)
   local trainNum = trainInput:size(1)

   -- concatenate
   local concatInput = torch.clone(torch.cat(trainInput, testInput, 1))
   local concatTarget = torch.clone(torch.cat(trainTarget, testTarget, 1))

   -- shuffle
   local shuffle = torch.randperm(concatInput:size(1))
   concatInput = concatInput:index(1, shuffle:long())
   concatTarget = concatTarget:index(1, shuffle:long())

   local size = concatInput:size(1)
   return concatInput[{{1,trainNum},{},{},{}}], concatTarget[{{1,trainNum}}], 
          concatInput[{{trainNum+1,size},{},{},{}}], concatTarget[{{trainNum+1, size}}] 
end

-- extract k items from each class 
function util.convert(target, map)
   local map = map or {}

   local cnt = 1
   local newTarget = torch.Tensor(target:size()):typeAs(target)
   for i=1,target:size(1) do
      if map[target[i]] == nil then
         map[target[i]] = cnt 
         cnt = cnt + 1
      end 

      newTarget[i] = map[target[i]]
   end 

   return newTarget, map 
end


return util
