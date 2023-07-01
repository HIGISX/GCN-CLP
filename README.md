# GCN-Greedy: A Hybrid Framework  using Graph Convolutional Network and Greedy algorithm for Covering Location Problem
This repository contains code for the study "GCN-Greedy: A Hybrid Framework  using Graph Convolutional Network and Greedy algorithm for Covering Location Problem"
We use GCN network to solve location set covering problem (LSCP) and maximum covering location problem (MCLP) problems.The optimization of the loss function takes into account the characteristics of covering location problems.
Our model is more accurate and more efficient than classical heuristic algorithms.


<img src="img/framework.jpg" width="500">


## Usage
We ran our code using Python 3.8.0, PyTorch 1.12.1 and CUDA 11.7.
### Dependencies
Install all dependencies using pip.
```shell
pip install -f requirements.txt
```
### Graph dataset generation
Generate graph samples for lscp and mclp problem.
```shell
python ./generate_sample.py lscp <src sample path dir > <dst path dir >
python ./generate_sample.py mclp <src sample path dir > <dst path dir >
```
### Train
Train the model using this framework.
```shell
python main.py train  <problem type>  <train sample path>  <model save dir>  
<problem type> : lscp/mclp
```

### Test
Test the trained model using test dataset.
```shell 
python main.py solve  <problem type>  <test sample path>  --pretrained_weights <model save path>  
```
