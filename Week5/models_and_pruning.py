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
number_of_epochs = 800

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


# TESTING MODELS

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
history = model.fit(train_generator, batch_size=64, epochs=number_of_epochs, validation_data=validation_generator)

# print(history.history['val_accuracy'])
# print(history.history['loss'])

# Accuracy plot 
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,number_of_epochs,50),history.history['accuracy'][::50], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,number_of_epochs,50),history.history['val_accuracy'][::50], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Accuracy',fontsize=25)
plt.xticks(np.arange(0,number_of_epochs+50,50),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('accuracy_roab_bonito.jpg',transparent=False)
plt.close()

# Loss plot 
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,number_of_epochs,50),history.history['loss'][::50], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,number_of_epochs,50),history.history['val_loss'][::50], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.xticks(np.arange(0,number_of_epochs+50,50),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('loss_roab_bonito.jpg',transparent=False)
plt.close()

tensorflow.keras.models.save_model(model, "def_weights_roab.h5", include_optimizer=False)

# --------------------------------------
# WEIGHT PRUNING

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

end_step = np.ceil(400 / batch_size).astype(np.int32) * number_of_epochs

accuracies_pruning = []
accuracies_pruning_retraining = []
# Define model for pruning
prune_percentages=np.arange(0,1,0.1).astype('float32')
accuracy_without_pruning=None

for v in prune_percentages:
  print('pruning %:',v)
  pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(v,0)}
  model_pruning = model
  model_pruning.load_weights("def_weights_roab.h5")

  model_for_pruning = prune_low_magnitude(model_pruning, **pruning_params)

  model_for_pruning.compile(optimizer='adam',
                loss="categorical_crossentropy",
                metrics=['accuracy'])

  logdir = tempfile.mkdtemp()
  callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
  ]
  
  model_for_pruning.fit(train_generator,
  epochs=1, batch_size=batch_size, callbacks=callbacks)

  tensorflow.keras.models.save_model(model, "def_weights_pruning_"+str(v)+".h5", include_optimizer=False)

  _,accuracy=model_for_pruning.evaluate(validation_generator,callbacks=callbacks)
  accuracies_pruning.append(accuracy)
  print("Pruning test accuracy: {:.2f}%".format(accuracies_pruning[-1]))
  if accuracy_without_pruning is None:
    accuracy_without_pruning = accuracies_pruning[-1]

  model_for_pruning.fit(train_generator, validation_data=validation_generator,
  epochs=number_of_epochs, batch_size=batch_size, callbacks=callbacks)
  _, model_for_pruning_accuracy = model_for_pruning.evaluate(validation_generator)
  print("Pruning + retraining test accuracy: {:.2f}%".format(model_for_pruning_accuracy * 100))

  accuracies_pruning_retraining.append(model_for_pruning_accuracy)
  # save model
  model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
  print('params:',model_for_pruning.count_params())

accuracy_loss_pruning = ((np.array(accuracies_pruning)-accuracy_without_pruning)/accuracy_without_pruning) * 100
accuracy_loss_pruning_retraining = ((np.array(accuracies_pruning_retraining)-accuracy_without_pruning)/accuracy_without_pruning) * 100

max_y=max(np.max(accuracy_loss_pruning),np.max(accuracy_loss_pruning_retraining))
min_y=min(np.min(accuracy_loss_pruning),np.min(accuracy_loss_pruning_retraining))

# Weight Pruning plot
plt.figure(figsize=(10,10),dpi=150)
plt.plot(prune_percentages*100,accuracy_loss_pruning, marker='o', color='orange',label='Pruning')
plt.plot(prune_percentages*100,accuracy_loss_pruning_retraining, marker='o', color='purple',label='Pruning + Retraining')
plt.plot(np.arange(0,110,10),np.zeros(11),linestyle=(0, (5, 5)), linewidth=1.5, color='black')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Weight Pruning',fontsize=25)
plt.xlim(0,100)
plt.ylim(min(-1,min_y-2),max(1,max_y+2))
plt.xticks(np.arange(0,110,10),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy loss (%)',fontsize=17)
plt.xlabel('Parameters pruned (%)',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('weight_pruning.jpg',transparent=False)

# _, pruned_keras_file = tempfile.mkstemp('.h5')
# tensorflow.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
# print('Saved pruned Keras model to:', pruned_keras_file)