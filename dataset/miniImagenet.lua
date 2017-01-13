local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'

local function loadSplit(splitFile)
   --[[
   Args:
      splitFile (string): path to split file
   --]]
   local classes = { }

   local fid = io.open(splitFile, 'r')
   local i = 1
   for line in fid:lines() do
      if i > 1 then
         local parsedLine = string.split(line, ',')
         local filename = parsedLine[1]
         local class = parsedLine[2]
         if not classes[class] then
            classes[class] = { }
         end
         _.push(classes[class], filename)
      end
      i = i + 1
   end
   fid:close()

   return classes
end

local function processor(ind, opt, input)
   if not threadInitialized then
      _ = require 'moses'
      local nn = require 'nn'
		local sys = require 'sys' 
      local image = require 'image'

      local pre = nn.Sequential() 
      pre:add(nn.Reshape(1, opt.imageDepth, opt.imageHeight, opt.imageWidth))
      pre:add(nn.MulConstant(1.0 / 255, true))

      if opt.train then
         pre:training()
      else
         pre:evaluate()
      end

      if opt.cuda then
         require 'cunn'
         pre:insert(nn.Copy('torch.ByteTensor', 'torch.CudaTensor'), 1)
         pre:cuda()
      else
         pre:insert(nn.Copy('torch.ByteTensor', 'torch.FloatTensor'), 1)
         pre:float()
      end

      function loadImage(path)
         local img = image.load(path, 3, 'byte')
         return image.scale(img, opt.imageWidth, opt.imageHeight) 
      end
	
      function preprocessImages(images)
         return torch.cat(_.map(images, function(i, v)
            return pre:forward(v):clone()
         end), 1)
      end

      threadInitialized = true
   end

   local class = opt.classes[ind]

   -- construct all image urls
	local urls = _.map(opt.imageFiles[class], function(i, v)
      return opt.dataDir .. '/' .. v
   end)	

   -- shuffle them and separate into train & test
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
	
   local metadata = {
      class = class,
      supportExamples = preprocessImages(supportImages),
      evalExamples = preprocessImages(evalImages)
   }

   collectgarbage()
   collectgarbage()

   return input, metadata
end

local function getData(opt)
   opt.dataCacheDir = opt.dataCacheDir or sys.fpath()

	local imageZipFile = 'images.zip'
	local imagesDir = 'images'
   local splitDir = paths.concat(opt.dataCacheDir, 'miniImagenet')
   local splits = {'train', 'val', 'test'}
   local splitFiles = { }
   _.each(splits, function(i, split)
      splitFiles[split] = split .. '.csv'
   end)
   local requiredFiles = _.append({imageZipFile}, _.values(splitFiles))
	
   if not paths.dirp(splitDir) then
      paths.mkdir(splitDir)
   end
 
	-- unzip images if necessary
	if not paths.dirp(paths.concat(splitDir, imagesDir)) then 
		print('unzipping: ' .. paths.concat(splitDir, imageZipFile))
		os.execute(string.format('unzip %s -d %s', paths.concat(splitDir, 
         imageZipFile), splitDir))
	end
	
	print('data dir: ' .. paths.concat(splitDir, imagesDir))
	local miniImagenetDataDir = paths.concat(splitDir, imagesDir)

   -- prepare datasets
   local ret = { }
   _.each(splits, function(i, split)
      -- class => image filename mapping
      local imageFiles = loadSplit(paths.concat(splitDir, splitFiles[split]))
      local classes = _.sort(_.keys(imageFiles))

      -- construct a dataset over class indices
      local ds = Dataset(torch.range(1, #classes))
      local get, size = ds.sampledBatcher({
         inputDims = { 1 },
         batchSize = opt.nClass[split],
         samplerKind = opt.episodeSamplerKind,
         processor = processor,
         cuda = opt.cuda,
         processorOpt = {
            dataDir = miniImagenetDataDir,
				imageFiles = imageFiles,
            classes = classes,
            nSupportExamples = opt.nSupportExamples,
            nEvalExamples = opt.nEvalExamples,
            classSamplerKind = opt.classSamplerKind,
            imageDepth = opt.imageDepth,
            imageHeight = opt.imageHeight,
            imageWidth = opt.imageWidth,
            resizeType = opt.resizeType,
            pre = opt.pre,
            train = _.contains({'train'}, split),
            cuda = opt.cuda
         }
      })

      ret[split] = {
         get = get,
         size = size
      }
   end)
 
   return ret
end

return getData
