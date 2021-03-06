# You Only Match Once: Point Cloud Registration with Rotation-equivariant Descriptors

In this paper, we propose a novel local descriptor-based framework, called You Only Hypothesize Once (YOHO), for the registration of two unaligned point clouds. In contrast to most existing local descriptors which rely on a fragile local reference frame to gain rotation invariance, the proposed descriptor achieves the rotation invariance by recent technologies of group equivariant feature learning, which brings more robustness to point density and noise. Meanwhile, the descriptor in YOHO also has a rotation equivariant part, which enables us to estimate the registration from just one correspondence hypothesis. Such property reduces the searching space for feasible transformations, thus greatly improves both the accuracy and the efficiency of YOHO. Extensive experiments show that YOHO achieves superior performances with much fewer needed RANSAC iterations on four widely-used datasets, the 3DMatch/3DLoMatch datasets, the ETH dataset and the WHU-TLS dataset.

## News
- 2021.9.1 Paper is accessible on arXiv.[paper](https://arxiv.org/abs/2109.00182)
- 2021.8.29 The code of the PointNet backbone YOHO is released, which is poorer but highly generalizable.
- 2021.7.6 The code of the YOHO is released. [Code](https://github.com/HpWang-whu/YOHO), [Project page](https://hpwang-whu.github.io/YOHO/)

## Performance 
<img src="README.assets/sendpix1.jpg" alt="sendpix1" style="zoom:50%;" />   

## Requirements

Here we offer the PointNet backbone YOMO thanks to the [Spinnet](https://github.com/QingyongHu/SpinNet)'s training codes, so the Spinnet requirements need to be met:

- Ubuntu 16.04 or higher
- CUDA 11.1 or higher
- Python v3.6 or higher
- Pytorch v1.7 or higher
- Pointnet2_ops

## Installation

Create the anaconda environment:

```
conda create -n pn_yomo python=3.7
conda activate pn_yomo
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-11.1
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

KNN build:

```
cd knn_search/
python setup.py build_ext --inplace
cd ..
```



## Data Preparation

We need the 3DMatch dataset (Train, Val) .

For the preparation of the Trainset of YOHO-PN, download [trainset](https://drive.google.com/file/d/1PrkSE0nY79gOF_VJcKv2VpxQ8s7DOITg/view?usp=sharing) Spinnet offering firstly.

Please place the trainset to ./data following Spinnet.

We offer the origin test datasets containing the point clouds (.ply) and keypoints (.txt, 5000 per point cloud) here [3dmatch/3dLomatch](https://drive.google.com/file/d/1UzGBPce5VspD2YIj7zWrrJYjsImSEc-5/view?usp=sharing), [ETH](https://drive.google.com/file/d/1hyurp5EOzvWGFB0kOl5Qylx1xGelpxaQ/view?usp=sharing) and [WHU-TLS](https://drive.google.com/file/d/1QjlxIVMQPinNWt5LKhtaG9TTo2j3TGs_/view?usp=sharing).

Please place the data to ```./data/origin_data``` for organizing the data structure as:

- data
  - origin_data
    -  3dmatch
      - sun3d-home_at-home_at_scan1_2013_jan_1
          - Keypoints
          - PointCloud
    - ETH
      - wood_autumn
        - Keypoints
        - PointCloud
    - WHU-TLS
      - Park
        - Keypoints
        - PointCloud

## Train

To prepare the trainset of YOHO, run:
```
cd backbone
python YOHO_trainset.py
cd ..
```
The trainset of YOHO will be placed to ```data/YOHO_PN/Trainset``` and ```data/YOHO_PN/valset```.


The training process of YOHO is two-stage, you can run with by the commands sequentially:

```
python Train.py --Part PartI
python Train.py --Part PartII
```

We also offer the pretrained models in ./model/PartI_train and ./model/PartII_train.

## Test on the 3dmatch/3dLomatch dataset

With the TestData downloaded above, the test on 3DMatch and 3DLoMatch can be done by:

- Prepare testset

```
cd backbone
python YOHO_testset.py --dataset 3dmatch --r 0.3
cd ..
```

- Eval the results:

```
python Test.py --Part PartI  --max_iter 1000 --dataset 3dmatch   --ransac_d 0.07 #YOHO-C on 3DMatch
python Test.py --Part PartI  --max_iter 1000 --dataset 3dLomatch --ransac_d 0.07 #YOHO-C on 3DLoMatch
python Test.py --Part PartII --max_iter 1000 --dataset 3dmatch   --ransac_d 0.09 #YOHO-O on 3DMatch
python Test.py --Part PartII --max_iter 1000 --dataset 3dLomatch --ransac_d 0.09 #YOHO-O on 3DLoMatch
```

where PartI is yoho-c and PartII is yoho-o, max_iter is the ransac times, PartI should be run first. All the results will be placed to ```./data/YOHO_PN```.


## Generalize to the ETH dataset

With the TestData downloaded above, without any refinement of the model trained on the indoor 3DMatch dataset, the generalization result on the outdoor ETH dataset can be got by:

- Prepare the testset

```
cd backbone
python YOHO_testset.py --dataset ETH --r 1
cd ..
```

- Eval the results:

```
python Test.py --Part PartI  --max_iter 1000 --dataset ETH  #YOHO-C on ETH
python Test.py --Part PartII --max_iter 1000 --dataset ETH  #YOHO-O on ETH
```
All the results will be placed to ```./data/YOHO_PN```.


## Generalize to the WHU-TLS dataset

With the TestData downloaded above, without any refinement of the model trained on the indoor 3DMatch dataset, the generalization result on the outdoor large scale TLS dataset WHU-TLS can be got by:

- Prepare the testset

```
cd backbone
python YOHO_testset.py --dataset WHU-TLS --r 6
cd ..
```

- Eval the results:

```
python Test.py --Part PartI  --max_iter 1000 --dataset WHU-TLS  --ransac_d 1 --tau_2 0.5 --tau_3 1  #YOHO-C on WHU-TLS
python Test.py --Part PartII --max_iter 1000 --dataset WHU-TLS  --ransac_d 1 --tau_2 0.5 --tau_3 1  #YOHO-O on WHU-TLS
```
All the results will be placed to ```./data/YOHO_PN```.


## Related Projects

We thanks greatly for the Spinnet, PerfectMatch, Predator and WHU-TLS for the training code and the datasets.

- [Spinnet](https://github.com/QingyongHu/SpinNet)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet) 
- [Predator](https://github.com/overlappredator/OverlapPredator) 
- [WHU-TLS](https://www.sciencedirect.com/science/article/pii/S0924271620300836)
