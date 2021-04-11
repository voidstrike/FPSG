# FPG: Ongoing Project

This repository contains the experimental PyTorch implementation of the paper:

**Few-shot Single Image Point Cloud Reconstruction: A Prototypical Approach**

## Introduction
Reconstructing point clouds from images would be extremely beneficial to practical applications, such as robotics, automated vehicles, and Augmented Reality. Many recent deep learning frameworks rely on a large amount of labeled training data to tackle this problem. However, in real-world scenarios, we usually have copious 2D images and are deficient in 3D shapes. Additionally, current available 3D data covers only a limited amount of classes, which further restricts the model from generalizing to novel classes.
## Intuition

PENDING

## Getting Started
### Installation

1. Clone this repo:
```
git clone https://github.com/voidstrike/FPG.git
```

2. Install the dependencies:
* Python 3.6
* CUDA 11.0
* G++ or GCC5
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.2.0
* [Scikit-learn](https://scikit-learn.org/stable/index.html) 0.24.0
* [Pymesh](https://github.com/PyMesh/PyMesh) 1.0.2

3. Compile CUDA kernel for CD/EMD loss (If you have trouble with this step, please refer to [PointFlow](https://github.com/stevenygd/PointFlow) for more information.)
```
cd src/metrics/pytorch_structural_losses/
make clean
make
```

4. Alternatively, using the Chamfer Distance from [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) repo. UPDATING

### Download Datasets

[ModelNet](https://modelnet.cs.princeton.edu/) dataset can be downloaded from [ModelNet_views](https://drive.google.com/file/d/19_aSXKe2xdOCw4_jEXjJcCUrHGl-HlFF/view?usp=sharing) and [ModelNet_pcs](https://drive.google.com/file/d/1XAVg8iZrOyE02cZxGdY1f880A1KBKZuu/view?usp=sharing).
Unzip the downloaded file to your preferred directory.

The 2D projections of ModelNet are from [MVCNN](https://github.com/suhangpro/mvcnn)

### Run Experiments

- Prepare the dataset (you will get two files: `modelnet_train.txt` and `modelnet_test.txt`,  `extra_file/` will contain numbers of auxiliary files )
```
python3 generate_dataset.py --img_path IMG_PATH --pc_path PC_PATH --dataset modelnet
```

- Train a model 
```
# General
CUDA_VISIBLE_DEVICES=X python3 trainNetwork.py --config_path TRAIN_CONFIG --test_path TEST_CONFIG --pc_encoder_path PRETRAINED_ENCODER --n_way 1 --n_shot [1-32] --num_cluster 1 --aggreage mean --name EXP_NAME

# Concrete
CUDA_VISIBLE_DEVICES=0 python3 trainNetwork.py --config modelnet_train.txt --test_path modelnet_test.txt --pc_encoder_path ../checkpoint/pretrain_pointnet/pretrained_pcencoder_pointnet.pt --n_way 1 --n_shot 32 --n_query 5 --num_cluster --epoch 400 --lr 0.0001 --aggregate mean --name modelnet_mean
```

- There are more hyper parameter options, please refer to the source code for more detail
- **Specifically**, make sure a). `n_way=1`; b). `num_cluster=1` when `aggregate=mean/mask_s` c). `num_cluster=n_shot` and `--num_slaves=1` when `aggregate=full/mask_m`.

- Please remember to modify the CUDA device number X, TRAIN_CONFIG, TEST_CONFIG and PRETRAINED_ENCODER accordingly.
- Eval a model (Actually, the evaluation code will be executed at the end of the training) -- PENDING

## Important Note
It's an experimental code and some code are hard-coded.
For example:

`src/datasets/modelnet.py` L110  -- **Modified the index number based on your file path**


## Citation

If you use this code for your research, please consider cite our paper:
PENDING
