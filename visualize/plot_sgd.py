import matplotlib.pyplot as plt

if __name__ == '__main__':

   train = [58.3, 66.6, 95.8, 99.9, 95.8, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9,
      99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9, 99.9]
   validation = [40.0, 50.6, 53.3, 58.6, 56.0, 57.3, 72, 72, 70.6, 76, 73.3, 
      74.6, 73.3, 77, 74.6, 73.3, 74.6, 73.3, 73.3]

   epochs = list(xrange(19))
   plt.plot(epochs, train, label='train accuracy')
   plt.plot(epochs, validation, label='test accuracy')
   plt.legend(loc='upper right')
   plt.show()  
