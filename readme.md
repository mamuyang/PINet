# Readme--PINet

This is the our implementation for the paper: 

Muyang Ma, Pengjie Ren, Yujie Lin, Zhumin Chen, Jun Ma, de Rijke, Maarten (2019). $\pi$-Net: A Parallel Information-sharing Network for Shared-account Cross-domain Sequential Recommendations. SIGIR'19, Paris, France, July 21-25, 2019

We build and release a shared-account smart TV recommendation dataset HVIDEO to facilitate research for SCSR.

**Please cite our paper if you use the code or dataset. Thanks!**

# HVIDEO description

## Data description ##

HVIDEO is a smart TV dataset that contains 260k users watching logs from October 1st 2016  to June 30th 2017. The logs are collected on two platforms (the V-domain and the E-domain) 
from a well-known smart TV service provider.

The V-domain contains family video watching behavior including TV series, movies, cartoons,  talent shows and other programs. 
And the E-domain covers online educational videos based on textbooks from elementary to high school, as well as instructional videos on sports, food,  medical, etc. 

On the two platforms, we gather user behaviors, including which video is played, when a  smart TV starts to play a video, and when it stops playing the video, and how long the video
has been watched.

## Data statistics ##

We craw 13,714 overlapped user, which includes 16,407 items, 227,390 logs of V-domain and  3,380 items, 177,758 logs of E-domain.

We randomly divide the data sets into training set(75%), test set(10%), valid set(15%).

# Code description #

## version ##
Python 3.6

Tensorflow 1.12.0

## baseline ##
1. POP, Item-KNN, BPR-MF see: [url](https://github.com/hidasib/GRU4Rec)

2. Conet see: Code.zip/baseline/Conet, which is quoted from "Conet: Collaborative Cross Networks for Cross-Domain Recommendation"

3. VUI-KNN see: Code.zip/baseline/VUI-KNN/, preprocess.py is the pre-processing code that generates  the input data needed by code vui-knn.py.

4. NCF-MLP++ see: Code.zip/baseline/ncfmlp.py, which is quoted from "Neural Collaborative Filtering"

5. GRU4REC see: Code.zip/baseline/GRU4REC.py, which is implemented in Tensorflow

6. HGRU4REC see: Code.zip/baseline/HRNN.py, which is implemented in Tensorflow

## pinet code ##

1. PINET model see: Code.zip/PINET/PiNet.py

2. No SFU see: Code.zip/PINET/PiNet_WSFU.py

3. No SFUCTU see: Code.zip/PINET/PiNet_WSFUACTU.py
