import math as math
import matplotlib.pyplot as plt
import numpy as np
import sys
import torchfile
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
   fileName = sys.argv[1]
   nExamples = 20
   data = torchfile.load(fileName)

   mean = data[0]
   stdv = data[1] 
  
   examples = []
   for i in range(nExamples):
      examples.append(data[1+i+1])
   #examples = [data[2], data[3], data[4], data[5], data[6]] 
  
   x = np.array(range(1,len(mean[0])+1))
   f, axarr = plt.subplots(len(mean))
   #f.subplots_adjust(hspace=1.0)
   f.tight_layout()

   for i in range(len(mean)):
      #axarr[i].plot(x, mean[i], color='b')
      for j in range(nExamples):
         axarr[i].plot(x, examples[j][i], 'r-', alpha=0.3)
      
      axarr[i].set_title('Layer ' + str(i+1))
      axarr[i].set_ylim([0,1.1])
      axarr[i].xaxis.set_major_locator(MaxNLocator(integer=True))
      #axarr[i].set_xlabel('Number of Updates', fontsize=8)
      
      #axarr[i].fill(np.concatenate([x, x[::-1]]), \
      #  np.concatenate([mean[i] - (1.9600 * stdv[i]/math.sqrt(50)),
      #                 (mean[i] + (1.9600 * stdv[i]/math.sqrt(50)))[::-1]]), \
      #  alpha=.5, fc='b', ec='None', label='95% confidence interval') 
   plt.show()


   '''
   print(len(data))
   print(len(data[0]))   
   data = data[0][layer] 

   #mean = np.mean(data1)
   #std = np.std(data1)
   #print mean, std

   x = np.array(range(1,len(data)+1))
   print(x)
   print(data)

   plt.plot(x, data, marker='o', color='b')
   plt.fill
   plt.legend(loc='lower right')
   plt.show()
   '''

   # Plot the function, the prediction and the 95% confidence interval based on
   # the MSE
   '''
   pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
   pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
   pl.plot(x, y_pred, 'b-', label=u'Prediction')
   pl.fill(np.concatenate([x, x[::-1]]), \
           np.concatenate([y_pred - 1.9600 * sigma,
                          (y_pred + 1.9600 * sigma)[::-1]]), \
           alpha=.5, fc='b', ec='None', label='95% confidence interval')
   pl.xlabel('$x$')
   pl.ylabel('$f(x)$')
   pl.ylim(-10, 20)
   pl.legend(loc='upper left')

   pl.show()
   '''
