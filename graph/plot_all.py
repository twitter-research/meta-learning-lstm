import matplotlib.pyplot as plt
import sys

FILE1 = 'stdout_simpleConv_2layerLstm_noBN'
FILE2 = 'stdout_BN'

def readFile(fileName):
   iters = []
   train_vals = []
   oneShot_vals = []
   fiveShot_vals = []

   with open(fileName) as f:
      iterFound = False
      trainFound = False
      oneFound = False

      iteration = 0
      for line in f:
         arr = line.split()
         if len(arr) >= 5 and arr[0] == 'Dataset:' and arr[3] == 'Loss:':
            iteration = int(arr[1][:-1])
            iterFound = True     

         if iterFound and trainFound and oneFound and len(arr) >=4 and arr[1] == 'global' and arr[2] == 'accuracy:':
            val = arr[3]
            if val.find("%") > 0:
               val = val.replace('%', '')
            
            fiveShot_vals.append(float(val)) 
            iters.append(iteration)
            
            iterFound = False
            trainFound = False
            oneFound = False

         elif iterFound and trainFound and len(arr) >=4 and arr[1] == 'global' and arr[2] == 'accuracy:':
            val = arr[3]
            if val.find("%") > 0:
               val = val.replace('%', '')
            
            oneShot_vals.append(float(val))  
            oneFound = True

         elif iterFound and len(arr) >=4 and arr[1] == 'global' and arr[2] == 'accuracy:':
            val = arr[3]
            if val.find("%") > 0:
               val = val.replace('%', '')
            
            train_vals.append(float(val)) 
            trainFound = True 
         
         elif len(arr) >=4 and arr[0] == 'Training' and arr[len(arr) - 2] == 'Accuracy:':
            iteration = int(arr[2].replace(',', ''))
            iterFound = True
            
            val = arr[len(arr) - 1]
            if val.find("%") > 0:
               val = val.replace('%', '') 
                   
            train_vals.append(float(val)) 
            trainFound = True

   return iters, train_vals, oneShot_vals, fiveShot_vals 

if __name__ == '__main__':
   args = []
   for i in range(1, len(sys.argv)):
      iters, train_vals, oneShot_vals, fiveShot_vals = readFile(sys.argv[i])
      plt.plot(iters, train_vals, label=sys.argv[i] + ":train")
      plt.plot(iters, oneShot_vals, label=sys.argv[i] + ":one-shot")
      plt.plot(iters, fiveShot_vals, label=sys.argv[i] + ":five-shot")
         
   plt.legend(loc='lower right', prop={'size':6})
   plt.show()
