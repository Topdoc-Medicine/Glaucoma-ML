import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
from concurrent import futures
import threading
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)

train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])

batch_size = 16
num_classes = 2
epochs = 25
INPUT_SHAPE = (256, 256, 3)

logdir = os.path.join('.',
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet',
                                        input_shape=INPUT_SHAPE)
vgg.trainable = False
# Freeze the layers
for layer in vgg.layers:
    layer.trainable = False

base_vgg = vgg
base_out = base_vgg.output
pool_out = tf.keras.layers.Flatten()(base_out)
hidden1 = tf.keras.layers.Dense(512, activation='relu')(pool_out)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=base_vgg.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary()

print("Total Layers:", len(model.layers))
print("Total trainable layers:", sum([1 for l in model.layers if l.trainable]))
# print(len(train_imgs_scaled), len(train_labels_enc))
# print(len(val_imgs_scaled), len(val_labels_enc))

# Initiate the train and test generators with data Augumentation
sometimes = lambda aug: iaa.Sometimes(0.6, aug)
seq = iaa.Sequential([
                      iaa.GaussianBlur(sigma=(0 , 1.0)),
                      iaa.Sharpen(alpha=1, lightness=0),
                      iaa.CoarseDropout(p=0.1, size_percent=0.15),
                              sometimes(iaa.Affine(
                                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                                    rotate=(-30, 30),
                                                    shear=(-16, 16)))
                    ])


train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=seq.augment_image,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.000001)

#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15,
#                                              mode='auto', baseline=None, restore_best_weights=False)
callbacks = [reduce_lr, tensorboard_callback]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=callbacks
)

model.save('vgg.h5')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

model.save('CNN4Weights.h5')
