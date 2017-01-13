from gensim import corpora, models, utils, matutils
import scipy.io as sio
import time
import sys
import logging

CAT_FILE = '/usr/people/sachinr/Downloads/XMLDatasetRead/ReadData_Matlab/ft_mat'
EMBED_DIM = 5000

if __name__ == '__main__':
   logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

   # read matlab file
   ft_mat = sio.loadmat(CAT_FILE)['ft_mat']
   print('read matrix')
   print('shape: ', ft_mat.shape)
   sys.stdout.flush()

   # test gensim 
   start = time.time()
   corpus = matutils.Sparse2Corpus(ft_mat)
   lsi = models.LsiModel(corpus, num_topics=EMBED_DIM)
   end = time.time()
   print('execution time: ', end - start)   

   # embed example
   embedX = matutils.Corpus2dense(lsi[corpus[0]])
   print(embedX)
