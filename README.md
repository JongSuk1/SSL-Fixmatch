# Rethinking Semi Supervised Learning in Long Tailed Distribution: What Can We do?
This project is a vision part project conducted in KAIST Computer Science CS492 (H): Special Topics in CS < Deep Learning for Real-World Problems > 

We use KAIST-NAVER product class 265 dataset from Naver Shopping website.


## Semi-supervised learning with imbalance resolving method
You can train model on local GPU(8 GPU) and NSML.
If you train on local, default model set to EfficientNet-b4 and on NSML, EfficientNet-b0. 

Here are two script for run on local and NSML environment.


### FixMatch
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch"
```

### Smoothed-FixMatch (Label smoothing)
You can use this method by change **smooth** option.
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --smooth 1 --name smoothed_fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --smooth 1 --name smoothed_fixmatch"
```

### OverFixMatch (Oversampling)
You can use this method by change **parser** option.
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --parser 3 --name overfixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --parser 3 --name overfixmatch"
```

### FixMatch-cRT (Class ReTraining)
First, you should have train model in normal distribution of NAVER Fashion Dataset.

Then, you should enter the model path for argument. It reinitialize and train **only linear classifier** with balanced distribution
```bash
> python balance_classifier.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --pretrained ./runs/{FILENAME} --name fixmatch_crt

> nsml run -d fashion_eval -e balance_classifier.py -a "--gpu_ids 0,1,2,3,4,5,6,7 --batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --pretrained ./runs/{FILENAME} --name fixmatch_crt"
```

|  <center>Method</center> |  <center>Top 1 Acc.(Val)</center> |  <center>Top 1 Acc.(Test)</center> | <center>model</center> |
|:--------|:--------:|--------:|:--------:|
|Vanila | <center> **77.99** </center> |**90.74** |<a href="https://drive.google.com/open?id=1W69Xb077zoGirovWc8ls750hRpP6krg-">download</a></td>|
|Smoothed-FixMatch | <center>73.51 </center> |- ||
|OverFixMatch | <center>76.32 </center> |86.77 ||
|FixMatch-cRT | <center>- </center> |89.68|<a href="https://drive.google.com/open?id=1kwuw_PlhDEcr3YQAPf4NDattHIW9VyMG">download</a></td>|

## Inference
### Submit local model
You should enter **sharable url path and filename** in code(**submit.py**). we use google drive to get sharable link.

If you want to change test batch size, change batch_size argument value in _infer function 
```bash
nsml run -d fashion_eval -e submit.py -a "--local 1"
nsml submit {SESSION NAME} best
```

### Submit NSML model
```bash
> nsml run -d fashion_eval -e submit.py -a "--session {SESSION NAME} --checkpoint {CHECKPOINT NAME}"
> nsml submit {SESSION NAME} best
```

Also you can submit session directly, but if you want change some inference method, use this option.

## Check Distribution of Data
```bash
python CountLabel.py
python CountUnlabel.py
```
