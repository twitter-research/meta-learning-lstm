import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys

params = ['W', 'bF', 'cI', 'WF', 'bI', 'WI']

def readFileParam(fileName, param):
   iters = []
   vals = []
   dataset = 'Dataset:'
   parameter = 'Parameter'
   
   d = 0
   is_param = False

   with open(fileName) as f:
      for line in f:
         arr = line.split()
      
         if dataset in arr:
            d = arr[arr.index(dataset) + 1]
            d = d.replace(',', '')
   
         if parameter in arr:
            is_param = True

         if d > 0 and is_param and param in arr:
            iters.append(d)
            v = arr[arr.index(param) + 2]
            v = v.replace(',', '')

            vals.append(v)
            d = 0    
            is_param = False

   return iters, vals

def readFileRatio(fileName, param):
   iters = []
   vals = []
   dataset = 'Dataset:'
   parameter = 'Update/Parameter'
   
   d = 0
   is_param = False

   with open(fileName) as f:
      for line in f:
         arr = line.split()
      
         if dataset in arr:
            d = arr[arr.index(dataset) + 1]
            d = d.replace(',', '')
   
         if parameter in arr:
            is_param = True

         if d > 0 and is_param and param in arr:
            iters.append(d)
            v = arr[arr.index(param) + 2]
            v = v.replace(',', '')

            vals.append(v)
            d = 0    
            is_param = False

   return iters, vals



if __name__ == '__main__':
   args = []
   for param in params:
      iters, vals = readFileParam(sys.argv[1], param)
      plt.plot(iters, vals, label=param)

   fontP = FontProperties()
   fontP.set_size('small')
   plt.legend(prop = fontP, loc=9, bbox_to_anchor=(0.5, 0), ncol=3)
   #plt.plot(*args)
   plt.show()
