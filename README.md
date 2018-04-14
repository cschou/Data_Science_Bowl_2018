# Data Science Bowl 2018

This is my work on the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018). Though the challenge has ended, more improvements are still in progress.

## Task
Identify all nuclei in a microscopy image taken in various contions(types of microscropes, lighting, staining, magnification...etc). Beyond segmentation accuracy, the evaluation metric puts more emphasis on instance detection accuracy, i.e., the rate of true positives. Hence, this is essentially an instance segmentation problem.

## Model
The Unet model used basically follows [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), implemented in Keras

## Usage
### Install Kaggle API and download datasets
```bash
  pip install kaggle
  kaggle competitions download -c data-science-bowl-2018
```



