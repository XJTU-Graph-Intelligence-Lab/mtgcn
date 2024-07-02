# MTGNN

## Brief Description of Code Structure

1. pretrain： auxiliary model
   
2. src：mutil-track model


#### Environmental Configuration

``` pip install -r requirements.txt ```


#### Instruction for Node Classification Experiment

```python mutil_stage_train.py --dataset='dataname' --a=0.9 --dr=0.5 --lr=0.01 --layer_num=64 ```