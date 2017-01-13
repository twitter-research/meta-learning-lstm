from gensim import corpora, models, utils, matutils
from scipy.sparse import find, csc_matrix, lil_matrix
import numpy as np
import scipy.io as sio
import time
import sys
import logging
import cPickle as pickle
import h5py

TRAIN_FT = '/usr/people/sachinr/Downloads/XMLDatasetRead/ReadData_Matlab/AmazonCat13K/train_ft_mat.mat'
TRAIN_LBL = '/usr/people/sachinr/Downloads/XMLDatasetRead/ReadData_Matlab/AmazonCat13K/train_lbl_mat.mat' 
TEST_FT =  '/usr/people/sachinr/Downloads/XMLDatasetRead/ReadData_Matlab/AmazonCat13K/test_ft_mat.mat'
TEST_LBL = '/usr/people/sachinr/Downloads/XMLDatasetRead/ReadData_Matlab/AmazonCat13K/test_lbl_mat.mat'

META_TRAIN = [0, 5000]
META_VAL = [5000, 10000]
META_TEST = [10001, 13329]   

META_TRAIN_MAX = 900000
META_VAL_MAX = 150000

meta_train_dict = dict()
meta_val_dict = dict()
meta_test_dict = dict()

EMBED_DIM = 2000

'''
1. Read train and test
2. Use meta-set splits to determine size of matrices for meta-train, meta-validation, and meta-test 
   and create respective data
3. Run LSA on meta-train data
   a. save LSA computation
4. Use params of LSA to run on meta-validation and meta-test data
5. Save data as hdf5 files to read in torch 
'''

if __name__ == '__main__':
   # Read sparse matrices corresponding to features+lbls for train+test
   train_ft = csc_matrix.transpose(sio.loadmat(TRAIN_FT)['ft_mat'])
   train_lbl = csc_matrix.transpose(sio.loadmat(TRAIN_LBL)['lbl_mat'])
   test_ft = csc_matrix.transpose(sio.loadmat(TEST_FT)['ft_mat'])
   test_lbl = csc_matrix.transpose(sio.loadmat(TEST_LBL)['lbl_mat'])

   print('train_ft', train_ft.shape)
   print('train_lbl', train_lbl.shape)
   print('test_ft', test_ft.shape)
   print('test_lbl', test_lbl.shape)
   print("\n")

   #subset = 1000
   #train_ft = train_ft[0:subset]
   #train_lbl = train_lbl[0:subset]
   #test_ft = test_ft[0:subset]
   #test_lbl = test_lbl[0:subset]   

   # Determine size of meta-set matrices; (iterate through train+test labels and add to appropiate dicts)
   cnt = 0
   for data in [train_lbl, test_lbl]: 
      for i in range(0, data.shape[0]):
         blah, classes, ones = find(data[i])
         if len(meta_train_dict.keys()) < META_TRAIN_MAX and len(filter(lambda cl : cl >= META_TRAIN[0] and cl <= META_TRAIN[1], classes)) > 0:
            meta_train_dict[cnt] = True
         elif len(meta_train_dict.keys()) < META_VAL_MAX and len(filter(lambda cl : cl >= META_VAL[0] and cl <= META_VAL[1], classes)) > 0:
            meta_val_dict[cnt] = True
         elif len(filter(lambda cl : cl >= META_TEST[0] and cl <= META_TEST[1], classes)) > 0: 
            meta_test_dict[cnt] = True
         elif len(filter(lambda cl : cl >= META_TRAIN[0] and cl <= META_TRAIN[1], classes)) > 0:
            meta_train_dict[cnt] = True
         elif len(filter(lambda cl : cl >= META_VAL[0] and cl <= META_VAL[1], classes)) > 0:
            meta_val_dict[cnt] = True
         '''
         for cl in classes:
            if cl >= META_TRAIN[0] and cl <= META_TRAIN[1]: meta_train_dict[cnt] = True
            elif cl >= META_VAL[0] and cl <= META_VAL[1]: meta_val_dict[cnt] = True
            else: meta_test_dict[cnt] = True
         '''

         cnt += 1

   meta_train_size = len(meta_train_dict.keys())
   meta_val_size = len(meta_val_dict.keys())
   meta_test_size = len(meta_test_dict.keys())
   print('meta-train size', meta_train_size)
   print('meta-val size', meta_val_size)
   print('meta-test size', meta_test_size)
   print("\n")   
   sys.stdout.flush()

   # Build sparse matrices and dict lbls for each set
   featuresNum = train_ft.shape[1]
   lblNum = train_lbl.shape[1] 
   meta_train_ft = csc_matrix((meta_train_size, featuresNum)).tolil()
   meta_train_lbl = dict()
   meta_val_ft = csc_matrix((meta_val_size, featuresNum)).tolil()
   meta_val_lbl = dict()
   meta_test_ft = csc_matrix((meta_test_size, featuresNum)).tolil()
   meta_test_lbl = dict()

   # Iterate through train+test and assign appropriately
   cnt = 0; meta_train_cnt = 0; meta_val_cnt = 0; meta_test_cnt = 0
   for ft,lbl in [ (train_ft, train_lbl), (test_ft, test_lbl) ]:
      for i in range(0, ft.shape[0]):
         blah, classes, vals = find(lbl[i])
         if cnt in meta_train_dict:
            meta_train_ft[meta_train_cnt] = ft[i]  
            meta_train_lbl[meta_train_cnt] = filter(lambda cl : cl >= META_TRAIN[0] and cl <= META_TRAIN[1], classes) 
            meta_train_cnt += 1
         if cnt in meta_val_dict: 
            meta_val_ft[meta_val_cnt] = ft[i]   
            meta_val_lbl[meta_val_cnt] = filter(lambda cl : cl >= META_VAL[0] and cl <= META_VAL[1], classes)  
            meta_val_cnt += 1 
         if cnt in meta_test_dict: 
            meta_test_ft[meta_test_cnt] = ft[i]
            meta_test_lbl[meta_test_cnt] = filter(lambda cl : cl >= META_TEST[0] and cl <= META_TEST[1], classes) 
            meta_test_cnt += 1

         cnt += 1
 
   assert meta_train_size == meta_train_cnt, "%d vs %d" % (meta_train_size, meta_train_cnt)
   assert meta_val_size == meta_val_cnt,     "%d vs %d" % (meta_val_size, meta_val_cnt)
   assert meta_test_size == meta_test_cnt,   "%d vs %d" % (meta_test_size, meta_test_cnt)
 
   # save matrices for checkpointing   
   pickle.dump( meta_train_ft, open('meta_train_ft.p', 'wb'))
   pickle.dump( meta_train_lbl, open('meta_train_lbl.p', 'wb'))
   pickle.dump( meta_val_ft, open('meta_val_ft.p', 'wb'))
   pickle.dump( meta_val_lbl, open('meta_val_lbl.p', 'wb'))
   pickle.dump( meta_test_ft, open('meta_test_ft.p', 'wb'))
   pickle.dump( meta_test_lbl, open('meta_test_lbl.p', 'wb'))

   # Run LSA on meta-train matrix
   sys.stdout.flush()
   logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   train_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_train_ft))
   lsi = models.LsiModel(train_corpus, num_topics=EMBED_DIM) 
   lsi.save('lsi_model.gensim')
   
   # Apply LSA to meta-val and meta-train matrices 
   meta_train_ft_emb = np.transpose(matutils.corpus2dense(lsi[train_corpus], EMBED_DIM))
   val_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_val_ft))
   meta_val_ft_emb = np.transpose(matutils.corpus2dense(lsi[val_corpus], EMBED_DIM))
   test_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_test_ft))
   meta_test_ft_emb = np.transpose(matutils.corpus2dense(lsi[test_corpus], EMBED_DIM))

   print('embedded matrices')
   print('train_ft_emb', meta_train_ft_emb.shape)
   print('val_ft_emb', meta_val_ft_emb.shape)
   print('test_ft_emb', meta_test_ft_emb.shape)
    
   # Save all matrices to hdf5 
   print(meta_train_lbl[0])
   print(meta_train_lbl[1])
   print(meta_train_lbl[len(meta_train_lbl)-1])
   meta_hdf5_files = ['meta_train.h5', 'meta_val.h5', 'meta_test.h5']
   for idx, (data, lbls) in enumerate([(meta_train_ft_emb, meta_train_lbl), (meta_val_ft_emb, meta_val_lbl), (meta_test_ft_emb, meta_test_lbl)]): 
      with h5py.File(meta_hdf5_files[idx], 'w') as hf:
         hf.create_dataset('data', data=data)
         for i in range(len(lbls)):
            hf.create_dataset('labels ' + str(i), data=lbls[i]) 

