# Rethinking Semi Supervised Learning in Long Tailed Distribution: What Can We do?
This project is a vision part project conducted in KAIST Computer Science CS492 (H): Special Topics in Computer Science: <Deep Learning for Real-World Problems> 

We used KAIST-NAVER 265 classes dataset collected from NAVER Shopping website and want to say thank you to [NSML](https://ai.nsml.navercorp.com/) for providing GPU resourses.

## Semi-supervised learning with imbalance resolving method
You can train model on both on local and NSML.
Because of the limited provide resource of NSML, I recommend you to train on EfficientNet-b0 with NSML.
Regarding local training with EfficientNet-b4, we strongly recommend multi-GPU training. we used 8 GTX 2080 ti GPUs.

You can train the model with the code below:

### FixMatch
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --name fixmatch"
```

### Smoothed-FixMatch (With label smoothing)
You can use this method by changing the **smooth** option.
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --smooth 1 --name smoothed_fixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--batchsize 64 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03  --smooth 1 --name smoothed_fixmatch"
```

### OverFixMatch (Oversampling)
You can use this method by changing the **parser** option.
```bash
> python train_fixmatch.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --parser 3 --name overfixmatch

> nsml run -d fashion_eval -e train_fixmatch.py -a "--batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --parser 3 --name overfixmatch"
```

### FixMatch-cRT (Classifier re-training)
You should have pretrained model with the Dataset.
The script below reinitialize and train **the only linear classifier** with balanced distribution.
```bash
> python balance_classifier.py --gpu_ids 0,1,2,3,4,5,6,7 --batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --pretrained ./runs/{FILENAME} --name fixmatch_crt

> nsml run -d fashion_eval -e balance_classifier.py -a "--batchsize 32 --lambda-u 3 --mu 3 --threshold 0.85 --lr 0.03 --pretrained ./runs/{FILENAME} --name fixmatch_crt"
```

|  <center>Method</center> |  <center>Top 1 Acc.(Val)</center> |  <center>Top 1 Acc.(Test)</center> | <center>model</center> |
|:--------|:--------:|--------:|:--------:|
|Vanila | <center> **77.99** </center> |**90.74** |<a href="https://drive.google.com/open?id=1W69Xb077zoGirovWc8ls750hRpP6krg-">download</a></td>|
|Smoothed-FixMatch | <center>73.51 </center> |- ||
|OverFixMatch | <center>76.32 </center> |<center>86.77</center> ||
|FixMatch-cRT | <center>- </center> |<center>89.68</center>|<a href="https://drive.google.com/open?id=1kwuw_PlhDEcr3YQAPf4NDattHIW9VyMG">download</a></td>|

## Inference
### Submit local model
To upload the trained large models weights into NSML (e.g. EfficientNet-b4), you should enter **sharable url path and filename** in code(**submit.py**). we recommand you to use google drive link sharing.

If you want to change test batch size, change batch_size argument value in _infer function 
```bash
nsml run -d fashion_eval -e submit.py -a "--local 1"
nsml submit {SESSION NAME} best
```

### Submit NSML model
Plus, if you want to inference with the model trained on NSML, you can simply use the same command provided by NSML, or either, 

```bash
> nsml run -d fashion_eval -e submit.py -a "--session {SESSION NAME} --checkpoint {CHECKPOINT NAME}"
> nsml submit {SESSION NAME} best
```

## Check Distribution of Data
```bash
python CountLabel.py
python CountUnlabel.py
```

