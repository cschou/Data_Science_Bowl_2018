import os
import cv2
import random
import sys
import warnings
import numpy as np
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import guess_spatial_dimensions, gray2rgb
from keras.utils import Progbar
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Setting seed for reproducability

# Data Path
TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'
#TEST_PATH = 'data/stage2_test/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Function read train images and mask return as nump array
def read_train_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    if os.path.isfile("train_img.npy") and os.path.isfile("train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load("train_img.npy")
        Y_train = np.load("train_mask.npy")
        return X_train,Y_train
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                        preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask
        a.update(n)
    np.save("train_img",X_train)
    np.save("train_mask",Y_train)
    return X_train,Y_train

# Function to read test images and return as numpy array
def read_test_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('\nGetting and resizing test images ... ')
    sys.stdout.flush()
    if os.path.isfile("test_img.npy") and os.path.isfile("test_size.npy"):
        print("Test file loaded from memory")
        X_test = np.load("test_img.npy")
        sizes_test = np.load("test_size.npy")
        return X_test,sizes_test
    b = Progbar(len(test_ids))
    for n, id_ in enumerate(test_ids):
        path = TEST_PATH + id_
#        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = imread(path + '/images/' + id_ + '.png')

        if (guess_spatial_dimensions(img) == 2):
            img = gray2rgb(img)
        else:
            img = img[:,:,:IMG_CHANNELS]

        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        b.update(n)
    np.save("test_img",X_test)
    np.save("test_size",sizes_test)
    return X_test,sizes_test

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
def mask_to_rle(preds_test_upsampled):
    new_test_ids = []
    rles = []
    b = Progbar(len(test_ids))
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
        b.update(n)
    return new_test_ids,rles

# load and preprocess train images
def preprocess_train_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    if os.path.isfile("train_img.npy") and os.path.isfile("processed_train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load("train_img.npy")
        Y_train = np.load("processed_train_mask.npy")
        return X_train, Y_train
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH + id_
        img = cv2.imread(path + '/images/' + id_ + '.png')
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
        kernel = np.ones((3,3),dtype=np.uint8)
        count = 0
        masks = []
        dilate_masks = []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
            mask_dilated = cv2.dilate(mask_, kernel, iterations=1)
            masks.append(mask_)
            dilate_masks.append(mask_dilated)

        mask1 = np.stack(masks, axis=-1)
        total_mask = np.amax(mask1, axis=-1)
        mask2 = np.stack(dilate_masks, axis=-1)
        num_nonzeros = np.count_nonzero(mask2, axis=-1)
        mask_overlap = np.zeros((num_nonzeros.shape), dtype=np.bool)
        mask_overlap[num_nonzeros > 1] = True 
        total_mask[mask_overlap > 0] = False


        Y_train[n] = np.expand_dims(total_mask, axis=-1)
        a.update(n)
    np.save("train_img",X_train)
    np.save("processed_train_mask.npy",Y_train)
    return X_train,Y_train

def plot_images(images, masks, num_images=0):
    n_img = 6
    fig, m_axs = plt.subplots(2, n_img, figsize = (12, 4))
    d = zip(images[1:, ...], masks[1:, ...])
    
    for (img, label), (c_im, c_lab) in zip(d, m_axs.T):
#        c_im.imshow(img)
        c_lab.imshow(np.squeeze(img))
        c_im.axis('off')
        c_im.set_title('Microscope')
        
        c_lab.imshow(np.squeeze(label))
#        c_lab.imshow(label)
        c_lab.axis('off')
        c_lab.set_title('Labeled')

    plt.show()

if __name__ == '__main__':
#    x,y = read_train_data()
#    x,y = read_test_data()
    x1, y1 = preprocess_train_data()
    print(x1.shape)
    print(y1.shape)
    plot_images(x1, y1)
