from data_util import read_train_data,read_test_data,prob_to_rles,mask_to_rle,resize, preprocess_train_data
from keras.preprocessing import image
from model import get_unet, dice_coef
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='path to the reused model')
args = parser.parse_args()


epochs = 1
seedin = 41

# get train_data
#train_img,train_mask = load_train_data()
train_img,train_mask = preprocess_train_data()

# get test_data
test_img,test_img_sizes = read_test_data()

# train data augmentation
image_datagen = image.ImageDataGenerator(
                                         shear_range=0.2,
                                         rotation_range=180,
                                         zoom_range=0.2,
					 width_shift_range=0.1,
					 height_shift_range=0.1,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         fill_mode='constant',
                                         cval=0.
                                         )

mask_datagen = image.ImageDataGenerator(
                                        shear_range=0.2,
                                        rotation_range=180,
                                        zoom_range=0.2,
					width_shift_range=0.1,
					height_shift_range=0.1,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0.
                                        )

image_datagen.fit(train_img, augment=True, seed=seedin)
mask_datagen.fit(train_mask, augment=True, seed=seedin)
image_generator = image_datagen.flow(train_img, shuffle=True, batch_size=32, seed=seedin)
mask_generator = mask_datagen.flow(train_mask, shuffle=True, batch_size=32, seed=seedin)
train_generator = zip(image_generator, mask_generator)

img_batch, mask_batch = next(train_generator)
print(img_batch.shape)
print(mask_batch.shape)



checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', monitor='dice_coef', mode='max', verbose=1, save_best_only=True)
#earlystopper = EarlyStopping(monitor='loss', patience=5, verbose=1)

# get u_net model
if args.model is not None:
    print('Loading model: {}'.format(args.model))
    u_net = load_model(args.model, custom_objects={'dice_coef':dice_coef})
#    u_net.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
else:
    u_net = get_unet()

# fit model on train_data
print("\nTraining...")
u_net.fit_generator(train_generator, epochs=epochs, steps_per_epoch=len(train_img) / 32, callbacks=[checkpointer])

# Predict on test data
print("Predicting")
test_mask = u_net.predict(test_img,verbose=1)
u_net.save('last.h5')

# Create list of upsampled test masks
test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                       (test_img_sizes[i][0],test_img_sizes[i][1]), 
                                       mode='constant', preserve_range=True))


test_ids,rles = mask_to_rle(test_mask_upsampled)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

print("\nData saved")
