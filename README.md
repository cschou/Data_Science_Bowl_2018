# Data Science Bowl 2018

This is my work on the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018). Though the challenge has ended, more improvements are still in progress.

## Task
Identify all nuclei in a microscopy image taken in various conditions(types of microscropes, lighting, staining, magnification...etc). Beyond segmentation accuracy, the evaluation metric puts more emphasis on instance detection accuracy, i.e., the rate of true positives. Hence, this is essentially an instance segmentation problem.

## Model
The Unet model used basically follows [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), implemented in Keras.
![alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

## Usage
### Install Kaggle API and download datasets
```bash
  pip install kaggle
  kaggle competitions download -c data-science-bowl-2018
```
Unzip stage1_train.zip and stage1_test.zip in a folder named 'data'
### Train the model
```python
  python train.py
```

### To Do
The original paper used a weighted cross-entropy scheme, which is not implemented here. It seems to be crucial to performance of the whole system.
