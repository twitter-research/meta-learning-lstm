local _ = require 'moses'
local Dataset = require 'dataset.Dataset'

local util = { }

local function augmentRotations(imageFiles)
   local ret = { }
   local rots = { 'rot000', 'rot090', 'rot180', 'rot270' }

   for k,v in pairs(imageFiles) do
      for i,rot in ipairs(rots) do
         ret[k .. '/' .. rot] = v
      end
   end

   return ret
end

function util.loadOmniglotSplit(splitFile, dataDir)
   --[[
   Args:
      splitFile (string): path to split file
   --]]
   local classes = { }

   local fid = io.open(splitFile, 'r')
   for class in fid:lines() do
      classes[class] = { }

      for file in paths.iterfiles(paths.concat(dataDir, class)) do
         _.push(classes[class], paths.concat(dataDir, class, file))
      end
   end
   fid:close()

	return augmentRotations(classes)
end

return util
