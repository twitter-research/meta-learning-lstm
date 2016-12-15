require 'pl'
_ = require 'moses'

local train =  {1, 8500}
local val   =  {8501, 10001}
local test  =  {10002, 13330}

-- return lbl, sparse tensor
function readExample(str)
   local arr = str:split(' ')
   local lbls = _.map(arr[1]:split(','), function(i,v) return tonumber(i) end)
   
   local tensorArr = {}
   for i=2,#arr do
      local e = arr[i]:split(':')
      table.insert(tensorArr, {e[1], e[2]})
   end

   return lbls, torch.Tensor(tensorArr):float()
end

function readFile(file)

   local cnt = 1
   local nPts, nFeatures, nLabels

   local input = {}
   local target = {} 

   for line in io.lines(file) do
      if cnt == 1 then
         local l = line:split(' ')
         nPts = tonumber(l[1])
         nFeatures = tonumber(l[2])
         nLabels = tonumber(l[3])  
      else
         local lbls, tensor = readExample(line)
         table.insert(target, lbls)
         table.insert(input, tensor)
      end

      cnt = cnt + 1
   end

   return input, target
end

--local fileTrain = 'AmazonCat/amazonCat_train.txt' 
local fileTest = 'AmazonCat/amazonCat_test.txt'

--local trainInput, trainTarget = readFile(fileTrain)
local testInput, testTarget = readFile(fileTest)
--print("nTrain", #trainInput, #trainTarget)
print("nTest", #testInput, #testTarget)

local trainSet = {}
local valSet = {}
local testSet = {}

function getClass(input, target, class)
   local selInput = {}
   local selTarget = {}
   for i=1,#target do 
      if _.contains(target[i], class) then
         table.insert(selInput, input[i])
         table.insert(selTarget, target[i])
      end
   end

   return selInput, selTarget
end

for i=train[1],train[2] do
   local selInput, selTarget = getClass(testInput, testTarget, i)
   trainSet[i] = {selInput, selTarget}
end

print(trainSet[1])
