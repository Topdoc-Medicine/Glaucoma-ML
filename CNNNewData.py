import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from imgaug import augmenters as iaa
import datetime
import os

img_width, img_height = 256, 256
input_shape = (img_width, img_height, 3)

train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
batch_size = 16
epochs = 25

inp = tf.keras.layers.Input(shape=input_shape)

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                               activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat = tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)

out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary()

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

logdir = os.path.join('.',
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=callbacks
)

model.save("model5.h5")
