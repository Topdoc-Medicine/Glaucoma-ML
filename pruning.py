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
from sklearn.preprocessing import LabelEncoder
from tensorflow_model_optimization.sparsity import keras as sparsity

# Load the serialized model
h5file =  "vgg.h5"

with h5py.File(h5file,'r') as fid:
     loaded_model = tf.keras.models.load_model(fid)
     
train_data_dir = "data/train"
validation_data_dir = "data/validation"
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])
nb_validation_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])
batch_size = 16
epochs = 25
     
logdir = './log'
end_step = np.ceil(1.0 * nb_train_samples / batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]

# new_pruned_model.fit(train_imgs_scaled, train_labels_enc,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           callbacks=callbacks,
#           validation_data=(val_imgs_scaled, val_labels_enc))
#
# score = new_pruned_model.evaluate(val_imgs_scaled, val_labels_enc, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

final_model = sparsity.strip_pruning(new_pruned_model)
final_model.summary()
final_model.save('finalPrunedWeights.h5')
