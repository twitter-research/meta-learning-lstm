local _ = require 'moses'
local c = require 'trepl.colorize'
local Dataset = require 'dataset.Dataset'

local util = require 'cortex-core.projects.research.metalearn.data.util'

local remoteDataDir = '/user/cortex/metalearn-sachinr/imagenet-local'

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
      local imagecore = require 'imagecore'
      local nn = require 'nn'
      local prenn = require 'prenn'
		local sys = require 'sys' 

      local pre = nn.Sequential()
      if opt.pre then
         assert(not opt.resizeType, 'resizeType is only used if there is no pre')
         pre:add(opt.pre(opt))
      else
         pre:add(prenn.Image(opt.imageDepth,
                             opt.imageHeight,
                             opt.imageWidth,
                             opt.resizeType))
      end

      pre:add(nn.Reshape(1, opt.imageDepth, opt.imageHeight, opt.imageWidth))
      pre:add(nn.MulConstant(1.0 / 255, true))

      if opt.train then
         pre:training()
      else
         pre:evaluate()
      end

      if opt.cuda then
         require 'cunn'
         require 'cudnn'
         pre:cuda()
      else
         pre:float()
      end

		function loadImage(path)
         return imagecore.loadFile(path)
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
	--[[local urls = _.map(opt.imageFiles[class], function(i, v)
      return opt.imageHost .. '/' .. v
   end)--]]

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
	
	local ret = { }

   if not paths.dirp(splitDir) then
      paths.mkdir(splitDir)
   end

   -- check which files we are missing
   local filesToFetch = _.select(requiredFiles, function(i, baseFile)
      return not paths.filep(paths.concat(splitDir, baseFile))
   end)

   -- fetch if necessary
   if #filesToFetch > 0 then
      print(c.green '==>' .. ' fetching miniImagenet files')
      _.each(filesToFetch, function(i, baseFile)
			print(c.green '==>' .. ' fetching ' .. baseFile)
			local timer = torch.Timer()
         -- retrieve from HDFS
         local remotePath = paths.concat(remoteDataDir, baseFile)
         local localPath = paths.concat(splitDir, baseFile)
         util.fetchHDFSfile(remotePath, localPath)

			print(c.green '==>' .. ' time to fetch: ' .. timer:time().real .. ' s')
      end)
		print(c.green '==>' .. 'finished fetching files')
   else
      print(c.green '==>' .. ' found miniImagenet files')
   end

	-- unzip images
	if not paths.dirp(paths.concat(splitDir, imagesDir)) then 
		print('unzipping: ' .. paths.concat(splitDir, imageZipFile))
		os.execute(string.format('unzip %s -d %s', paths.concat(splitDir, imageZipFile), splitDir))
	end
	
	print('data dir: ' .. paths.concat(splitDir, imagesDir))
	local miniImagenetDataDir = paths.concat(splitDir, imagesDir)
   --local imageHost = loadImageHost(paths.concat(splitDir, 'meta.csv'))

   -- prepare datasets
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

   ret.buildDatasets = function(sample)
      return util.buildDatasets(sample, opt.batchSize, opt.exampleSamplerKind, opt.cuda)
   end

   return ret
end

return getData
