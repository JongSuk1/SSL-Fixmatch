## Rethinking Semi Supervised Learning in Long Tailed Distribution: What Can We do?
### This project is a vision part project conducted in KAIST Computer Science CS492 (H): Special Topics in CS < Deep Learning for Real-World Problems > 
We use KAIST-NAVER product class 265 dataset from Naver Shopping website.


## Semi-supervised learning with imbalance resolving method
You can train model on local GPU(8 GPU) and NSML.
If you train on local, default model set to EfficientNet-b4 and on NSML, EfficientNet-b0. 

Here are two script for run on local and NSML environment.


### FixMatch
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch"

### Smoothed-FixMatch (Label smoothing)
You can use this method by change smooth option.
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  **--smooth 1** --name smoothed_fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  **--smooth 1** --name smoothed_fixmatch"


### OverFixMatch (Oversampling)
You can use this method by change parser option.
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 **--parser 3** --name overfixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 **--parser 3** --name overfixmatch"


### FixMatch-cRT (Class ReTraining)
First, you should have train model in normal distribution of NAVER Fashion Dataset.

Then, you should enter the model path in the code. It reinitialize and train **only linear classifier** with balanced distribution
> python **balance_classifier.py** --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch_crt

> nsml run -d fashion_eval -e **balance_classifier.py** -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch_crt"




## Inference
### Submit local model
You should enter **sharable url path and filename** in code(**submit.py**). we use google drive to get sharable link.

If you want to change test batch size, change batch_size argument value in _infer function 
> nsml run -d fashion_eval -e submit.py -a "--local 1"

> nsml submit {SESSION NAME} best


### Submit NSML model
> nsml run -d fashion_eval -e submit.py -a "--session {SESSION NAME} --checkpoint {CHECKPOINT NAME}"

> nsml submit {SESSION NAME} best

Also you can submit session directly, but if you want change some inference method, use this option.
