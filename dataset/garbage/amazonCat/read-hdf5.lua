require 'hdf5'
_ = require 'moses'

metaHdf5Files = {'meta_train.h5', 'meta_val.h5', 'meta_test.h5'}
FOLDERS = {'train', 'val', 'test'}
PREFIX = 'example_'

for i=1,#metaHdf5Files do 
   if not paths.dirp(FOLDERS[i]) then
      paths.mkdir(FOLDERS[i])
   end   

   local classMap = {}

   print("processing: " .. metaHdf5Files[i])
   local f = hdf5.open(metaHdf5Files[i], 'r')
   local dataList = f:read('data'):all():type(torch.getdefaulttensortype())
   
   for j=1,dataList:size(1) do
      lbls = f:read('labels ' .. j-1):all()
      
      for k=1,lbls:size(1) do
         if classMap[lbls[k]] == nil then
            classMap[lbls[k]] = {}
         end
         table.insert(classMap[lbls[k]], j)
      end
  
      torch.save( paths.concat(FOLDERS[i], PREFIX .. j .. '.th'), {input=dataList[j]:float(), target=lbls}) 
   end   

   local finalClassMap = {}
   _.each(classMap, function(k,v) 
      if #v >= 20 then
         finalClassMap[k] = v
      end
   end)

   print('classMap size: ' .. #(_.keys(finalClassMap)))
   torch.save( paths.concat(FOLDERS[i], 'classMap.th'), {classMap=finalClassMap})
   --torch.save(metaHdf5Files[i]:gsub(".h5", ".th"), {input=dataList[i], target=lblsList[i], classMap=classMap[i]})
end

