local Dataset = torch.class('Dataset')

function Dataset:__init(lopt)
   self.X = lopt.x
   self.Y = lopt.y
   self.batchIdx = 1
   
   self.batchSize = lopt.batchSize or self.X:size(1)
   self.shuffle = lopt.shuffle or true
   self:shuffleData()
end

function Dataset:shuffleData()
   if self.shuffle then
      local shuffle = torch.randperm(self.X:size(1))
      self.X = self.X:index(1, shuffle:long())
      self.Y = self.Y:index(1, shuffle:long())     
   end
end

function Dataset:get()
   local idx = (self.batchIdx - 1) * self.batchSize + 1  
   self.batchIdx = self.batchIdx + 1      

   local input = self.X[{{idx, idx+self.batchSize-1}}]:clone()
   local target = self.Y[{{idx, idx+self.batchSize-1}}]:clone()

   if self.batchIdx > self:size() then
      self.batchIdx = 1
      self:shuffleData()
   end

   return {
      input = input,
      target = target
   }
end

function Dataset:size()
   return self.X:size(1)/self.batchSize
end

return Dataset
