#!/usr/bin/env torch

-- options
local opt = require 'cortex-core.libs.opt' [[
   Train a one-shot network
   
   Experiment config file (overrides options below):
   --task               (default -)                               path to config file for task 
   --data               (default -)                               path to config file for data
   --model              (default -)                               path to config file for model
   --test               (default -)                               path to config file for test details

   Architecture options:
   --nIn                (default 28)                              input dimensionality 

   Optimization options:
   --model              (default model.conv-net)                  classification model 
   --learner            (default model.sgd)                       learning-to-learn model
   --learningRate       (default 0.0001)                          learning rate
   --batchSize          (default 32)                              batch size
   --nEpochs            (default 10)                              max number of epochs

   Dataset options:
   --dataset            (default oneShotLSTM.dataset.dataset)     dataset reader to use
   --datasetHdfsRoot    (default users/cortex/one-shot-lstm/)     path to hdfs data
   --dataName           (default omniglot)                        name of dataset directory
   --trainFile          (default train.th)                        name of training set file
   --validationFile     (default validation.th)                   name of validation set file
   --testFile           (default test.th)                         name of test set file
]]

-- run
local result = require('cortex-core.projects.research.oneShotLSTM.train.train')(opt)
print('Task: ' .. opt.task .. '; Data: ' .. opt.data .. '; Model: ' .. opt.learner)
print('Test Accuracy: ')
print(result)

