# Self-Contrastive Learning Results Reproduction

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Parameters for Pretraining](#parameters-for-pretraining)
* [Contact](#contact)

## Installation
We experimented with eight RTX 3090 GPUs and CUDA version of 11.3.   
Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage
To train, run the selfcon_train script, which deoes the pretraining and linear evalutaion together. it takes the dataset config path as an argument

```bash
./scripts/selfcon_train.sh configs/datasets/flowers.yml
```



### Parameters for Pretraining
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Default: `resnet50`. |
| `dataset`      | Dataset to use. Options:  `cifar10`, `cifar100`, `cub`, `flowers`, `aircraft`. |
| `method`      | Pretraining method. Options:  `Con`, `SupCon`, `SelfCon`. |
| `lr` | Learning rate for the pretraining. Default: `0.5` for the batch size of 1024. |
| `temp` | Temperature of contrastive loss function. Default: `0.07`. |
| `precision` | Whether to use mixed precision. Default: `False`. |
| `cosine` | Whether to use cosine annealing scheduling. Default: `False`. |
| `selfcon_pos` | Position where to attach the sub-network. Default: `[False,True,False]` for ResNet architectures. |
| `selfcon_arch` | Sub-network architecture. Options: `resnet`, `vgg`, `efficientnet`, `wrn`. Default: `resnet`. |
| `selfcon_size` | Block numbers of a sub-network. Options: `fc`, `small`, `same`. Default: `same`. |
| `multiview` | Whether to use multi-viwed batch. Default: `False`. |
| `label` | Whether to use label information in a contrastive loss. Default: `False`. |

## Contact
* Sangmin Bae: bsmn0223@kaist.ac.kr
* Sungnyun Kim: ksn4397@kaist.ac.kr


## Notes 
* Multiview benefits small batch size 
<p align="center">
  <img src=image.png width="800">
</p>

* The loss can be very unstable. Need to tune and choose a good learning rate (I tried with 0.005)