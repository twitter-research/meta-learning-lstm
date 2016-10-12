import matplotlib.pyplot as plt
import sys

FILE1 = 'stdout_simpleConv_2layerLstm_noBN'
FILE2 = 'stdout_BN'

def readFile(fileName):
	iters = []
	vals = []

	with open(fileName) as f:
		iterFound = False
		iteration = 0
		for line in f:
			arr = line.split()
			if len(arr) >= 5 and arr[0] == 'Dataset:' and arr[3] == 'Loss:':
				iteration = int(arr[1][:-1])
				iterFound = True		

			if iterFound and len(arr) >=4 and arr[1] == 'global' and arr[2] == 'accuracy:':
				val = arr[3]
				if val.find("%") > 0:
					val = val.replace('%', '')
				
				vals.append(float(val))	
				iters.append(iteration)
				iterFound = False	

	return iters, vals 

if __name__ == '__main__':
	args = []
	for i in range(1, len(sys.argv)):
		iters, vals = readFile(sys.argv[i])
		plt.plot(iters, vals, label=sys.argv[i])
			
	plt.legend(loc='lower right')
	plt.show()
