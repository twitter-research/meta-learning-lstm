local _ = require 'moses'
local util = require 'cortex-core.projects.research.oneShotLSTM.util.util'

local opt = {}
opt.cuda = false
opt.nExamples = 20
opt.nDepth = 1
opt.nIn = 28

opt.dataName = 'omniglot'
opt.homePath = 'cortex-core.projects.research.oneShotLSTM.'

opt.dataLoader = 'dataset.data-loader2' 

opt.nClasses = {train=20, val=20, test=20}
opt.nTrainShot = 5
opt.nTestShot = {1,5}
opt.nEval = 15

--opt.trainFull = true
opt.batchSize = 32

opt.nAllClasses = 4112 

require(opt.homePath .. 'dataset.dataset')
local dataOpt = { 
   cuda = opt.useCUDA,
   episodeSamplerKind = 'uniform',

   dataCacheDir = '/Users/sachinr/workspace/local/',

   nClass = opt.nClasses,
   nSupportExamples = math.max(opt.nTrainShot, math.max(unpack(opt.nTestShot))) ,
   nEvalExamples = opt.nEval,

   imageDepth = opt.nDepth,
   imageHeight = opt.nIn,
   imageWidth = opt.nIn,
   resizeType = 'scale',

   normalizeData = false
}

local function prepareDataset(split, sample, field, batchSize)
   local examples = torch.cat(_.map(sample.item, function(i, item)
      return item.extra[field]
   end), 1) 

   local classes
   if opt.trainFull and split == 'train' then
      classes = torch.cat(_.map(sample.item, function(i, item)
         return item.extra[field].new(item.extra[field]:size(1)):fill(item.url)
      end), 1)
   else
      classes = torch.cat(_.map(sample.item, function(i, item)
         return item.extra[field].new(item.extra[field]:size(1)):fill(i)
      end), 1)
   end

   local ds = Dataset({ x = examples, y = classes, batchSize=batchSize, shuffle=true})

   return ds 
end

local data = require('cortex-core.projects.research.metalearn.data.' .. opt.dataName)(dataOpt)

_.each(data, function(k,v)
   if type(data[k]) == 'table' and data[k].get then
      print("k: " .. k .. ": " .. opt.nClasses[k])
      data[k].createEpisode =
         function(lopt)
            local rawData = data[k].get()
            print(rawData)
            while #rawData.item <= 0 or (opt.nClasses[k] and #rawData.item < opt.nClasses[k]) do
               rawdata = data[k].get()
            end
            return prepareDataset(k, rawData, 'supportExamples', lopt.trainBatchSize), prepareDataset(k, rawData, 'evalExamples', lopt.testBatchSize)
         end
   end
end)

local display = require 'display'
local image = require 'image'

local trainSet, testSet = data.train.createEpisode({})
print('train')
local b = 1
while b <= trainSet:size() do
   local sample = trainSet:get()
   local input, target = util.extractK(sample.input, sample.target, 1, opt.nClasses.train)
   display.image(image.toDisplayTensor(input))
   display.image(image.toDisplayTensor(sample.input))
   print(b, input:size(), target)
   b = b + 1
end

print('test')
local b = 1
while b <= testSet:size() do
   local sample = testSet:get()
   display.image(image.toDisplayTensor(sample.input))
   print(b, sample.input:size(), sample.target)
   b = b + 1
end
