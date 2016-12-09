local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'

local util = require 'dataset.util'

local function processor(ind, opt, input)
   if not threadInitialized then
      _ = require 'moses'
      local nn = require 'nn'
		local sys = require 'sys' 
      local image = require 'image'

      local classMap = torch.load(paths.concat(opt.dataDir, 'classMap.th')).classMap
 
      local pre = nn.Sequential()   

      pre:add(nn.Reshape(1, opt.imageWidth))
      --[[pre:add(nn.MulConstant(1.0 / 255, true))
      --]]
      
      if opt.train then
         pre:training()
      else
         pre:evaluate()
      end

      if opt.cuda then
         require 'cunn'
         --require 'cudnn'
         pre:insert(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'), 1)
         pre:cuda() 
      else
         pre:float()
      end
      --]]

      function loadImage(path)
         local tensor = torch.load(path).input 
         return tensor 
      end

      function loadLbl(path)
         local lbl = torch.load(path).target
         return lbl
      end
	
      function preprocessImages(images)
         return torch.cat(_.map(images, function(i, v)
            return pre:forward(v):clone()
         end), 1)
      end

      function getExamples(class)
         return classMap[class]
      end

 
      threadInitialized = true
   end

   -- get class assigned
   local class = opt.classes[ind]
 
   -- construct all image url   
   local urls = _.map(getExamples(class), function(i, v)
      return paths.concat(opt.dataDir, 'example_' .. v .. '.th')
   end)	

   -- shuffle them
   urls = _.shuffle(urls)

   local supportUrls = _.first(urls, opt.nSupportExamples)
   local evalUrls = _.rest(urls, opt.nSupportExamples + 1)	
 
   -- filter down evaluation, if necessary
   if opt.nEvalExamples then
      evalUrls = _.first(evalUrls, opt.nEvalExamples)
   end		
 
   -- fetch images
   local supportImages = { }
   local supportLbls = {}	
   local evalImages = { }	
   local evalLbls = {}

	for i,url in ipairs(supportUrls) do
      _.push(supportImages, loadImage(url))
      _.push(supportLbls, loadLbl(url))
   end
   for i,url in ipairs(evalUrls) do
      _.push(evalImages, loadImage(url))
      _.push(evalLbls, loadLbl(url))
   end
 
   local supportExamples = preprocessImages(supportImages)
   local evalExamples = preprocessImages(evalImages)
   	
   local metadata = {
      class = class,
      supportExamples = supportExamples,
      supportLbls = supportLbls,
      evalExamples = evalExamples,
      evalLbls = evalLbls
   }

   collectgarbage()
   collectgarbage()

   return input, metadata
end

local function getData(opt)
   opt.dataCacheDir = opt.dataCacheDir or sys.fpath()

   local splitDir = paths.concat(opt.dataCacheDir, 'AmazonCat13K')
   local splits = {'train', 'val', 'test'}
  
   local ret = { }
 
   -- prepare datasets
   _.each(splits, function(i, split)
      local classMap = torch.load(paths.concat(paths.concat(splitDir, split), "classMap.th")).classMap
      local classes = _.sort(_.keys(classMap))

      -- construct a dataset over class indices
      local ds = Dataset(torch.range(1, #classes))
      local get, size = ds.sampledBatcher({
         inputDims = { 1 },
         batchSize = opt.nClass[split],
         samplerKind = opt.episodeSamplerKind,
         processor = processor,
         cuda = opt.cuda,
         poolSize = 1,
         processorOpt = {
            dataDir = paths.concat(splitDir, split),
            classes = classes,
            nSupportExamples = opt.nSupportExamples,
            nEvalExamples = opt.nEvalExamples,
            classSamplerKind = opt.classSamplerKind, 
            pre = opt.pre,
            train = _.contains({'train'}, split),
            cuda = opt.cuda,   
            imageDepth = opt.imageDepth,
            imageHeight = opt.imageHeight,
            imageWidth = opt.imageWidth,
         }
      })

      ret[split] = {
         get = get,
         size = size
      }
   end)

   ret.buildDatasets = function(sample)
      return util.buildDatasets(sample, opt.batchSize, opt.exampleSamplerKind, opt.cuda)
   end

   return ret
end

return getData
