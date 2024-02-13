import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
# from MixUpImageDataGenerator import *
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### HYPERPARAMETERS
train_set = 'MIT_small_train_1'
root_dir = './' + train_set

train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height=224
batch_size=4
epochs = 1000

### CREATE DATASET
train_datagen = ImageDataGenerator()

itr = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=400,
class_mode='categorical')
itr.reset()
X_train, y_train = itr.next()
print(X_train.shape, y_train.shape)

test_datagen = ImageDataGenerator()
itr = test_datagen.flow_from_directory(
test_data_dir,
target_size=(img_width, img_height),
batch_size=400,
class_mode='categorical')
itr.reset()
X_test, y_test = itr.next()
print(X_test.shape, y_test.shape)

train_datagen_tmp = ImageDataGenerator(
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=90,
  # rescale=1./255,
  # preprocessing_function=get_random_eraser(v_l=0, v_h=255)
  # featurewise_center=True,
  # featurewise_std_normalization=True
)

# train_datagen.fit(X_train)

test_datagen_tmp = ImageDataGenerator(
    # rescale=1./255,
    # featurewise_center=True,
    # featurewise_std_normalization=True
)

# test_datagen.fit(X_train)

# train_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=1, datagen=train_datagen_tmp)()


train_generator = train_datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])

validation_generator = test_datagen_tmp.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])




# model = keras.models.Sequential([
#   keras.layers.Conv2D(64 , 3, input_shape=(224, 224, 3)),
#   keras.layers.BatchNormalization(),
#   keras.layers.MaxPooling2D(pool_size=2),
#   keras.layers.Conv2D(32 , 3),
#   keras.layers.BatchNormalization(),
#   keras.layers.MaxPooling2D(pool_size=2),
#   keras.layers.Conv2D(16 , 3),
#   keras.layers.BatchNormalization(),
#   keras.layers.MaxPooling2D(pool_size=2),
#   # keras.layers.Flatten(),
#   keras.layers.GlobalAveragePooling2D(),
#   # keras.layers.Dense(1024, activation='relu'),
#   # keras.layers.Dropout(.3),
#   keras.layers.Dense(8, activation='softmax'),
# ])

# OUR SMALL SQUEEZENET
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, 3, strides=2, input_shape=[224, 224, 3]))
model.add(FireUnit(8, 16, 16))
model.add(keras.layers.Activation('relu'))
# model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Dense(8, activation='softmax'))



# model = keras.models.Sequential([
#   keras.layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=[224, 224, 3]),
#   keras.layers.MaxPooling2D(2),
#   keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#   keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#   keras.layers.MaxPooling2D(2),
#   keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#   keras.layers.MaxPooling2D(2),
#   keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#   keras.layers.GlobalAveragePooling2D(),
#   # keras.layers.Flatten(),
#   keras.layers.Dense(128, activation='relu'),
#   # keras.layers.Dropout(0.2),
#   keras.layers.Dense(64, activation='relu'),
#   # keras.layers.Dropout(0.2),
#   keras.layers.Dense(8, activation='softmax')
# ])

# TRAINING WITH DIFFERENT METRICS
print(model.summary())
print(model.count_params())
# sys.exit()
# opt = keras.optimizers.Adam(learning_rate=0.1)
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0)
opt = tf.keras.optimizers.Adadelta(learning_rate=0.1)
# opt = keras.optimizers.Adam(learning_rate=0.0001)
# opt = tf.keras.optimizers.Adamax(learning_rate=0.1)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
history = model.fit(train_generator, batch_size=1, epochs=epochs, validation_data=validation_generator)

# history = model.fit_generator(generator=train_generator,
#                     steps_per_epoch=X_train.shape[0] // batch_size,
#                     validation_data=(X_test, y_test),
#                     epochs=epochs, verbose=1)

# print(history.history['val_accuracy'])
# print(history.history['loss'])

# SAVING RESULTS
list_of_files = glob.glob('./outputs/graphs/accuracy_mlp*.jpg') # * means all if need specific format then *.csv
if len(list_of_files) > 1:
  latest_file = max(list_of_files, key=os.path.getctime)
  latest_file = os.path.basename(latest_file)
  file_index = int(re.search(r'\d+', latest_file).group())
  file_index += 1
else:
  file_index = 1

offset = 100
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),history.history['accuracy'][::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),history.history['val_accuracy'][::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Accuracy',fontsize=25)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./outputs/graphs/accuracy_mlp'+str(file_index)+'.jpg',transparent=False)
plt.close()

plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),history.history['loss'][::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),history.history['val_loss'][::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.ylim(0, 4)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./outputs/graphs/loss_mlp'+str(file_index)+'.jpg',transparent=False)
plt.close()

list_of_files = glob.glob('./outputs/graphs/loss_mlp*.jpg') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
latest_file = os.path.basename(latest_file)
file_index = int(re.search(r'\d+', latest_file).group())
file_index += 1

plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),history.history['loss'][::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),history.history['val_loss'][::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./outputs/graphs/loss_mlp'+str(file_index)+'.jpg',transparent=False)
plt.close()

# plt.figure(figsize=(5,5))
# plt.plot(history.history['accuracy'], label='Training accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation accuracy')
# plt.title('Accuracy of MLP')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.xticks(np.arange(0,epochs+1,20))
# plt.tight_layout()
# plt.legend()
# plt.savefig('./outputs/graphs/accuracy_mlp'+str(file_index)+'.jpg')
# plt.close()

# plt.figure(figsize=(5,5))
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.title('Loss of MLP')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0, 4)
# plt.xticks(np.arange(0,epochs+1,20))
# plt.legend()
# plt.tight_layout()
# plt.savefig('./outputs/graphs/loss_mlp'+str(file_index)+'.jpg')
# plt.close()