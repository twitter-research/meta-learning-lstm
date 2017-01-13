import matplotlib.pyplot as plt
import sys

FILE1 = 'stdout_simpleConv_2layerLstm_noBN'
FILE2 = 'stdout_BN'

def readFile(fileName):
   iters = []
   losses = []

   with open(fileName) as f:
      for line in f:
         arr = line.split()
         if len(arr) >= 5 and arr[0] == 'Dataset:' and arr[3] == 'Loss:':
            iters.append(int(arr[1][:-1]))
            
            loss = arr[4]
            if loss.find(",") > 0:
               loss = loss.replace(',', '')
            losses.append(float(loss))
               

   return iters, losses

if __name__ == '__main__':
   args = []
   for i in range(1, len(sys.argv)):
      iters, losses = readFile(sys.argv[i])
      plt.plot(iters, losses, label=sys.argv[i])
      #args.append(iters)
      #args.append(losses)
      #args.append(readFile(sys.argv[i]))

   
   #iters1, losses1 = readFile(FILE1)
   #iters2, losses2 = readFile(FILE2)
   #plt.plot(iters1, losses1, iters2, losses2)
   plt.legend(loc='upper right')
   #plt.plot(*args)
   plt.show()
