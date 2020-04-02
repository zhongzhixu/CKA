# CNA

This repository provides a tensorflow implementation of the CNA model introduced in paper "A comorbidity network-aware modle for prognostic prediction".

CNA is a novel prognostic prediction model which exhibits superior performance on disease risk prediction while provides potential disease propagation routes in comorbidity network.

The general architecture of CNA is showed in the following.

![figure](https://github.com/zhongzhixu/CNADRP/blob/master/icon/figure1.jpg)

The data is provided by the Hospital Authority of Hong Kong the ethical approval UW11-495. The data can not be made available to others according to the Hospital Authority and the ethical approval. Instead, we provide some simulated cases in DATA folder.  

Part of the codes are referenced Dr. Wang HW's RIPPLENET (https://github.com/hwwang55/RippleNet). We thank them very much for sharing their codes.

## Files in the folder
DATA: data_sample.csv
kg_final.txt

data_loader.py
model.py
train.py
main.py

## Environment:
Python 3.6

Keras 2.2.4

TensorFlow 1.13.1

## Running the code

Clone all files to the local computer, and fill in appropriate path.
Then the programme works.

