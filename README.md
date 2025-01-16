# Multi-Track Message Passing: Tackling Oversmoothing and Oversquashing in Graph Learning via Preventing Heterophily Mixing

This repository contains the Code of the ICML’24 paper _Multi-Track Message Passing: Tackling Oversmoothing and Oversquashing in Graph Learning via Preventing Heterophily Mixing_.

## Brief Description of Code Structure

1. pretrain： auxiliary model
   
2. src：mutil-track model


#### Environmental Configuration

``` pip install -r requirements.txt ```


#### Instruction for Node Classification Experiment

```python mutil_stage_train.py --dataset='dataname' --a=0.9 --dr=0.5 --lr=0.01 --layer_num=64 ```


## Contact
Please contact liyu1998@stu.xjtu.edu.cn or peihongbin@xjtu.edu.cn if you have any questions.

## Cite
Please cite our papers if you use the model or this code in your own work:

```
@inproceedings{pei2024multitrack,
title={Multi-Track Message Passing: Tackling Oversmoothing and Oversquashing in Graph Learning via Preventing Heterophily Mixing},
author={Hongbin Pei and Yu Li and Huiqi Deng and Jingxin Hai and Pinghui Wang and Jie Ma and Jing Tao and Yuheng Xiong and Xiaohong Guan},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=1sRuv4cnuZ}
}

```
