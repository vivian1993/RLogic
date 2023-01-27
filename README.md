# RLoigc
## Introduction
The official Pytorch implementation of the paper [RLogic: Recursive Logical Rule Learning from Knowledge Graphs](https://dl.acm.org/doi/pdf/10.1145/3534678.3539421)

## KG Data:
* entities.txt: a collection of entities in the KG
* relations.txt: a collection of relations in the KG
* facts.txt: a collection of facts in the KG 
* train.txt: the model is trained to fit the triples in this data set
* valid.txt: create a blank file if no validation data is available
* test.txt: the learned ryles is evaluated on this data set for KG completion task

## Usage
For example, this command train a RLogic on family dataset using gpu 0
```
  python main.py --train --test --recur --get_rule --data family --topk 2400 --gpu 0
```
Each parameter means:
```
* --train
--test

```
