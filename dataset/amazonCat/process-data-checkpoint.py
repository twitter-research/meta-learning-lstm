from gensim import corpora, models, utils, matutils
from scipy.sparse import find, csc_matrix, lil_matrix
import numpy as np
import cPickle as pickle
import h5py

EMBED_DIM = 2000

if __name__ == '__main__':
   #meta_train_ft = pickle.load( open('meta_train_ft.p', 'rb'))
   #meta_train_lbl = pickle.load( open('meta_train_lbl.p', 'rb'))
   #meta_val_ft = pickle.load( open('meta_val_ft.p', 'rb'))
   #meta_val_lbl = pickle.load( open('meta_val_lbl.p', 'rb'))
   meta_test_ft = pickle.load( open('meta_test_ft.p', 'rb'))
   meta_test_lbl = pickle.load( open('meta_test_lbl.p', 'rb'))

   print('read pickle file')
   lsi = models.LsiModel.load('lsi_model.gensim')
   
   # Apply LSA to meta-val and meta-train matrices 
   #train_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_train_ft))
   #print('sparse 2 corpus')
   #meta_train_ft_emb = np.transpose(matutils.corpus2dense(lsi[train_corpus], EMBED_DIM))
   #val_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_val_ft))
   #meta_val_ft_emb = np.transpose(matutils.corpus2dense(lsi[val_corpus], EMBED_DIM))
   test_corpus = matutils.Sparse2Corpus(lil_matrix.transpose(meta_test_ft))
   meta_test_ft_emb = np.transpose(matutils.corpus2dense(lsi[test_corpus], EMBED_DIM))
   
   print('embedded matrices')
   #print('train_ft_emb', meta_train_ft_emb.shape)
   #print('val_ft_emb', meta_val_ft_emb.shape)
   print('test_ft_emb', meta_test_ft_emb.shape)
    
   # Save all matrices to hdf5 
   #print(meta_train_lbl[0])
   #print(meta_train_lbl[1])
   #print(meta_train_lbl[len(meta_train_lbl)-1])
   #meta_hdf5_files = ['meta_train.h5', 'meta_val.h5', 'meta_test.h5']
   meta_hdf5_files = ['meta_test.h5']
   for idx, (data, lbls) in enumerate([(meta_test_ft_emb, meta_test_lbl)]): 
      with h5py.File(meta_hdf5_files[idx], 'w') as hf:
         hf.create_dataset('data', data=data)
         for i in range(len(lbls)):
            hf.create_dataset('labels ' + str(i), data=lbls[i]) 

