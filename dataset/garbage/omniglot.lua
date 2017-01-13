local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'

local util = require 'cortex-core.projects.research.oneShotLSTM.dataset.util'

local function processor(ind, opt, input)
   if not threadInitialized then
      _ = require 'moses'
      local nn = require 'nn'
      local image = require 'image'
      local util = require 'cortex-core.projects.research.oneSHotLSTM.dataset.util'

      -- load mapping from classname to image path
      imageFiles = util.loadOmniglotSplit(opt.splitFile, opt.dataDir)

      -- construct nn module to preprocess each image
      assert(opt.imageDepth == 1, 'omniglot images are grayscale')
      local pre = nn.Sequential()
      -- image loads omniglot images as 3-channels (even when imageDepth = 1)
      -- in order to force them to be single-channel, we take the mean along dimension 1
      -- pre:add(nn.Sum(1, nil, true))
      pre:add(nn.Reshape(1, opt.imageDepth, opt.imageHeight, opt.imageWidth))
		
      if opt.normalizeData == nil or opt.normalizeData == true then 
			pre:add(nn.MulConstant(-1.0, true))
			pre:add(nn.AddConstant(1.0, true))
		else
			print("Not normalizing images!")
		end  

      if opt.cuda then
         require 'cunn'
         require 'cudnn'
         pre:insert(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'), 1)
         pre:cuda()
         
      else
         pre:insert(nn.Copy('torch.DoubleTensor', 'torch.FloatTensor'), 1)
         pre:float()
      end 

      function loadImage(path)
         local img = image.load(path)
         return image.scale(img, opt.imageWidth, opt.imageHeight) 
      end

		rotfns = {
			rot000 = function(img) return img end,
			rot090 = function(img) return image.rotate(img, math.pi / 2) end,
			rot180 = function(img) return image.rotate(img, math.pi) end,
			rot270 = function(img) return image.rotate(img, 3 * math.pi / 2) end
		}

      function preprocessImages(images, rot)
         return torch.cat(_.map(images, function(i, v)
            return pre:forward(rotfns[rot](v)):clone()
         end), 1)
      end

      threadInitialized = true
   end

   local class = opt.classes[ind]
   local urls = imageFiles[class]

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
   local evalImages = { }
   for i,url in ipairs(supportUrls) do
      _.push(supportImages, loadImage(url))
   end
   for i,url in ipairs(evalUrls) do
      _.push(evalImages, loadImage(url))
   end

   local rot = _.last(string.split(class, '/'), 1)[1]

   local metadata = {
      class = class,
      supportExamples = preprocessImages(supportImages, rot),
      evalExamples = preprocessImages(evalImages, rot)
   }

   collectgarbage()
   collectgarbage()

   return input, metadata
end

--[[
   Assumption is that either 
   (1) background+evaluation zip files exist in opt.dataCacheDir: 
         they will be unzipped and data will be placed as needed
   OR
   (2) omniglot/data/ exists in opt.dataCacheDir with all image files

   AND
   (1) omniglot/splits/ exists in opt.dataCache Dir with train, val, and test splits 
--]]
local function getData(opt)
   --opt.dataCacheDir = opt.dataCacheDir or sys.fpath()

   local ret = { }
   local omniglotDir = paths.concat(opt.dataCacheDir, 'omniglot/data')

   -- fetch data if necessary
   if not paths.dirp(omniglotDir) then
      print(c.green '==>' .. ' unzipping Omniglot data')
      paths.mkdir(omniglotDir)

      _.each({'background', 'evaluation'}, function(i,v)
         local baseDir = string.format('images_%s', v)
         local baseFile = string.format('%s.zip', baseDir)

         --[[ -- retrieve from HDFS
         util.fetchHDFSfile(paths.concat(remoteDataDir, baseFile),
                            paths.concat(opt.dataCacheDir, baseFile))
         --]]

         local baseFilePath = paths.concat(opt.dataCacheDir, baseFile) 
         assert(paths.filep(baseFilePath), string.format("%s not found", baseFilePath))
         
         -- extract
         os.execute(string.format('unzip %s -d %s',
                                  baseFilePath,
                                  omniglotDir))

         -- group together
         os.execute(string.format('mv %s/* %s/',
                                  baseFilePath,
                                  omniglotDir))
         os.execute('rmdir ' .. baseFilePath)
      end)
   else
      print(c.green '==>' .. ' found Omniglot data')
   end

   local splitDir = paths.concat(opt.dataCacheDir, 'omniglot/splits')
   local splits = {'train', 'val', 'test'}

  
   assert(paths.dirp(splitDir), string.format("%s not found", splitDir))
   _.each(splits, function(i,split)
      local splitPath = paths.concat(splitDir, split .. '.txt') 
      paths.filep(splitPath, string.format("%s not found", splitPath))
      end)
   print(c.green '==>' .. ' found Omniglot splits')

   -- prepare datasets
   _.each(splits, function(i, split)
      local imageFiles = util.loadOmniglotSplit(paths.concat(splitDir, split .. '.txt'), omniglotDir)
      local classes = _.sort(_.keys(imageFiles))
      print(c.green '==>' .. ' loaded ' .. #classes .. ' ' .. split .. ' classes')	

      -- construct a dataset over class indices
      local ds = Dataset(torch.range(1, #classes))
      local get, size = ds.sampledBatcher({
         inputDims = { 1 },
         samplerKind = opt.episodeSamplerKind,
         batchSize = opt.nClass[split],
         processor = processor,
         cuda = opt.cuda,
         processorOpt = {
				classes = classes,
            splitFile = paths.concat(splitDir, split .. '.txt'),
            dataDir = omniglotDir,
            nSupportExamples = opt.nSupportExamples,
            nEvalExamples = opt.nEvalExamples,
            classSamplerKind = opt.classSamplerKind,
            imageDepth = opt.imageDepth,
            imageHeight = opt.imageHeight,
            imageWidth = opt.imageWidth,
            resizeType = opt.resizeType,
            pre = opt.pre,
            cuda = opt.cuda,
				normalizeData = opt.normalizeData,
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
