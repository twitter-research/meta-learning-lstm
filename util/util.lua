
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

function util.boolize(x)
    if x == 'false' then
        x = false
    else
        x = true
    end
    return x
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

-- get k items from each class 
function util.extractK(input, target, k, nClasses)
   if k*nClasses == target:size(1) then
      return input, target
   end 
 
   local inputNew 
   if input:nDimension() == 4 then 
      inputNew = torch.Tensor(k*nClasses, input:size(2), input:size(3), input:size(4)):typeAs(input)
   elseif input:nDimension() == 2 then
      inputNew = torch.Tensor(k*nClasses, input:size(2)):typeAs(input)
   end
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

-- get random subset of items 
function util.getRandomSubset(input, target, num)
   local shuffle = torch.randperm(input:size(1))
   input = input:index(1, shuffle:long())
   target = target:index(1, shuffle:long())

   if input:nDimension() == 4 then  
      return input[{{1,num},{},{},{}}], target[{{1,num}}]
   elseif input:nDimension() ==2 then
      return input[{{1,num},{}}], target[{{1,num}}]
   end
end

function util.getBatch(input, target, idx, batchSize) 
   local x
   local y = target[{{idx,idx+batchSize-1}}]
   if input:nDimension() == 4 then
      x = input[{{idx,idx+batchSize-1},{},{},{}}]
   elseif input:nDimension() == 2 then
      x = input[{{idx,idx+batchSize-1},{}}]
   end

   return x,y
end 

return util
