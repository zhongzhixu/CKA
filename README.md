# CNA

This repository provides a tensorflow implementation of the CNA model introduced in paper "A comorbidity network-aware modle for prognostic prediction".

CNA is a novel prognostic prediction model which exhibits superior performance on disease risk prediction while provides potential disease propagation routes in comorbidity network.

The general architecture of CNA is showed in the following.

![figure](https://github.com/zhongzhixu/Dx2vec-for-Self-harm-prediction/blob/master/architecture_multi_input.jpg)

The data is provided by the Hospital Authority of Hong Kong the ethical approval UW11-495. The data can not be made available to others according to the Hospital Authority and the ethical approval. Instead, we provide some simulated cases in DATA folder.  

## Files in the folder
DATA: simulated cases

DX2vec.py

## Environment:
Python 3.6

Keras 2.2.4

TensorFlow 1.13.1

## Running the code

Clone all files to the local computer, fill appropriate path.

Note to split the study examples in DATA file into case group and control group.

Or, one can use all study samples as case group and control group simultaneously to just have a feeling of the code.
