# meta-learning-lstm
This repo contains the code for the following paper:
https://openreview.net/pdf?id=rJY0-Kcll 

## Dependencies
The following libaries are necessary:
# [torch-autograd][https://github.com/twitter/torch-autograd]
# [torch-ipc][https://github.com/twitter/torch-ipc] (use version from commit 'c1b2984c4c2dae085005d385996f4c0660173b27')
# [torch-Dataset][https://github.com/twitter/torch-dataset]
# [moses][https://github.com/Yonaba/Moses]

## Training
Splits corresponding to meta-training, meta-validation, and meta-testing are 
placed in `data/miniImagenet/`. Download corresponding imagenet images and
place in folder called `images` and place folder in `data/miniImagenet/`.

To train a model:
```
th run/train.lua --task [1-shot or 5-shot task] --data config.imagenet --model [model name]
```

For example, to run matching-nets:
```
th run/train.lua --task config.5-shot-5-class --data config.imagenet --model config.baselines.train-matching-net
```

And, to run LSTM meta-learner for 5-shot task:
```
th run/train.lua --task config.5-shot-5-class --data config.imagenet --model config.lstm.train-imagenet-5shot
```

## TODO 
* Include mini-ImageNet splits
