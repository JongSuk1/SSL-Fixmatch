### for supervised classification
##### without triplet and ranking loss
> python train_classification.py --gpu_ids 0 --trainfile kaist_naver_prod200k_class265_train01.txt --name your_model_name --lr 1e-4 --batchsize 128 --epochs 200 --lr 5e-3 --momentum 0.9

##### with triplet and ranking loss
> python train_classification.py --gpu_ids 0 --tripletmode --trainfile kaist_naver_prod200k_class265_train01.txt --name your_model_name --lr 1e-4 --batchsize 128 --epochs 200 --lr 5e-3 --momentum 0.9 --lossXent 1 --lossTri 0.5

---

### for semi-supervised classification
> python train_mixmatch.py --gpu_ids 0 --trainfile kaist_naver_prod200k_class265_train01.txt --batchsize 128 --lr 1e-3 --epochs 200 --momentum 0.9 --name your_model_name


---
### for test
> python test.py --gpu_ids 0 --reuse --name your_model_name 
