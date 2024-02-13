import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from models import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_model_optimization as tfmot
import time
import tempfile
from sklearn.metrics import accuracy_score
from sys import getsizeof
from tensorflow.keras.models import Sequential
from keras.models import load_model
import math
from random import sample


train_set = 'MIT_small_train_1'
root_dir = './' + train_set

train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height = 224
batch_size = 128
number_of_epochs = 2000


train_datagen = ImageDataGenerator(
  rescale=1./255,
  # featurewise_center=True,
  # featurewise_std_normalization=True
)

# train_datagen.fit(X_train)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    # featurewise_center=True,
    # featurewise_std_normalization=True
)

# test_datagen.fit(X_train)

train_generator = train_datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])

validation_generator = test_datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])

# print(train_generator)
# print(validation_generator)
# sys.exit()


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 3, strides=1, input_shape=[img_width, img_height, 3], padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(keras.layers.GlobalAveragePooling2D())
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(1000, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(250, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(8, activation='softmax'))


# tensorflow.keras.models.load_model(model, "def_weights_pruning_"+str(0.9)+".h5")

# print(model.summary())
# print(model.count_params())
# sys.exit()
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0)
opt = tf.keras.optimizers.Adadelta(learning_rate=0.1)
# opt = keras.optimizers.Adam(learning_rate=0.1)
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# opt = keras.optimizers.Adam(learning_rate=0.0001)
# opt = tf.keras.optimizers.Adamax(learning_rate=0.1)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
# history = model.fit(train_generator, batch_size=64, epochs=number_of_epochs, validation_data=validation_generator)

# print(history.history['val_accuracy'])
# print(history.history['loss'])

# tensorflow.keras.models.save_model(model, "def_weights_roab.h5", include_optimizer=False)

# --------------------------------------
### Retraining the model with 90% pruning previously saved

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

end_step = np.ceil(400 / batch_size).astype(np.int32) * number_of_epochs

v=0.9
print('pruning %:',v)
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(v,0)}
model_pruning = model
model_pruning.load_weights("def_weights_pruning_0.9.h5")

model_for_pruning = prune_low_magnitude(model_pruning, **pruning_params)

model_for_pruning.compile(optimizer='adam',
            loss="categorical_crossentropy",
            metrics=['accuracy'])

logdir = tempfile.mkdtemp()
callbacks = [
tfmot.sparsity.keras.UpdatePruningStep(),
tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

# model_for_pruning.fit(train_generator,
# epochs=1, batch_size=batch_size, callbacks=callbacks)

_,accuracy=model_for_pruning.evaluate(validation_generator,callbacks=callbacks)
# accuracies_pruning.append(accuracy)
print("Pruning test accuracy: {:.2f}%".format(accuracy))
# if accuracy_without_pruning is None:
# accuracy_without_pruning = accuracies_pruning[-1]

history = model_for_pruning.fit(train_generator, validation_data=validation_generator,
epochs=number_of_epochs, batch_size=batch_size, callbacks=callbacks)
_, model_for_pruning_accuracy = model_for_pruning.evaluate(validation_generator)
print("Pruning + retraining test accuracy: {:.2f}%".format(model_for_pruning_accuracy * 100))

# accuracies_pruning_retraining.append(model_for_pruning_accuracy)
# save model
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
print('params:',model_for_pruning.count_params())

tensorflow.keras.models.save_model(model, "def_weights_pruning_final_"+str(v)+".h5", include_optimizer=False)

# Accuracy plot for 90% Pruning
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,number_of_epochs,125),history.history['accuracy'][::125], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,number_of_epochs,125),history.history['val_accuracy'][::125], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Accuracy (Model with 90% pruning)',fontsize=25)
plt.xticks(np.arange(0,number_of_epochs+125,125),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('accuracy_pruning_90.jpg',transparent=False)
plt.close()

# Loss plot for 90% Pruning
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,number_of_epochs,125),history.history['loss'][::125], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,number_of_epochs,125),history.history['val_loss'][::125], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss (Model with 90% pruning)',fontsize=25)
plt.xticks(np.arange(0,number_of_epochs+125,125),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('loss_pruning_90.jpg',transparent=False)
plt.close()