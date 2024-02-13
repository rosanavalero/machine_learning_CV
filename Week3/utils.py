#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import wandb
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_contour
from optuna.visualization import plot_slice

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from wandb.keras import WandbCallback

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')


### From team 2

def get_files(IMG_SIZE=256):
  train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
  test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
  train_images_filenames = [n[16:] for n in train_images_filenames]
  test_images_filenames  = [n[16:] for n in test_images_filenames]
  train_labels = pickle.load(open('train_labels.dat','rb')) 
  test_labels = pickle.load(open('test_labels.dat','rb'))
  
  
  images_filenames = train_images_filenames.copy()
  images_filenames.extend(test_images_filenames)
  labels = train_labels.copy()
  labels.extend(test_labels)
  
  X = np.zeros((len(labels), 2), dtype='object')
  X[:, 0] = images_filenames
  X[:, 1] = labels
  
  clases = list(set(train_labels))
  
  train_images = []
  train_labels_final = []

  for filename, label in zip(train_images_filenames, train_labels):      
      ima=cv2.imread(filename)
      ima = cv2.resize(ima, (IMG_SIZE,IMG_SIZE))
      
      gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
      train_images.append(ima)

      class_id = clases.index(label)
      label_vector = np.zeros((len(clases)))
      label_vector[class_id] = 1
      train_labels_final.append(label_vector)

  test_images = []
  test_labels_final = []

  for filename, label in zip(test_images_filenames, test_labels):
      ima=cv2.imread(filename)
      ima = cv2.resize(ima, (IMG_SIZE,IMG_SIZE))
      
      gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
      test_images.append(ima)

      class_id = clases.index(label)
      label_vector = np.zeros((len(clases)))
      label_vector[class_id] = 1
      test_labels_final.append(label_vector)

  train_images = np.array(train_images)
  train_labels = np.array(train_labels_final)

  test_images = np.array(test_images)
  test_labels = np.array(test_labels_final)
  
  return X, train_images, test_images, train_labels, test_labels


def make(config, IMG_SIZE=256, phase='train'):       
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
    model.add(Dense(units=config.hidden, activation='relu',name='second', input_shape=(IMG_SIZE*IMG_SIZE*3,)))
    
    if phase=='test':
      model.add(Dense(units=config.classes, activation='linear'))
    else:
      model.add(Dense(units=config.classes, activation='softmax'))
      
    if config.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
    model.compile(loss=config.loss_function, optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    
    return model, optimizer
  
def normalize_data(train_images, test_images, IMG_SIZE=256, normalization='scaling'):
  train_images = train_images.reshape(train_images.shape[0], np.product(train_images.shape[1:]))
  test_images = test_images.reshape(test_images.shape[0], np.product(test_images.shape[1:]))
  
  if normalization == 'scaling':
    train_images_mean = np.mean(train_images, axis=0)
    train_images_std = np.std(train_images, axis=0)
    train_images_scaled = (train_images - train_images_mean) / train_images_std
    test_images_scaled = (test_images - train_images_mean) / train_images_std
  elif normalization == 'unit':
    train_images_scaled = train_images/255
    test_images_scaled = test_images/255
  
  train_images_scaled = train_images_scaled.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
  test_images_scaled = test_images_scaled.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
  
  return train_images_scaled, test_images_scaled


def histogramIntersection(M, N):

  K = np.zeros((M.shape[0], N.shape[0]), dtype=float)

  for j, _ in enumerate(N):
    for i, _ in enumerate(M):
      K[i][j] = np.sum(np.minimum(M[i], N[j]))

  return K


def create_model_deeper_wider(trial, IMG_SIZE):
  tf.keras.backend.clear_session()
    
  activation = trial.suggest_categorical("activation", ["relu", "selu", "elu", "swish"])
  depth = trial.suggest_int("depth", 1, 12)
  width = trial.suggest_int("width", 16, 2048)
  
  model = Sequential()
  model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
  model.add(Dense(units=width, activation='relu',name='second', input_shape=(IMG_SIZE*IMG_SIZE*3,)))
  for i in range(1,depth):
    model.add(Dense(units=width, activation='relu',name='second_'+str(i)))
  model.add(Dense(units=8, activation='softmax'))
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  
  trial.set_user_attr("parameters", model.count_params())
  # print(model.summary())
  
  return model

def objective(trial):
    IMG_SIZE = trial.suggest_categorical("image size", [8, 16, 32, 64, 128])
  
    batch_size = 8
    epochs = 2
    DATASET_DIR = 'MIT_split'
  
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            DATASET_DIR+'/train', 
            target_size=(IMG_SIZE, IMG_SIZE), 
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')  

    validation_generator = test_datagen.flow_from_directory(
            DATASET_DIR+'/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')
  
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()
    
    
    # Generate our trial model.
    model = create_model_deeper_wider(trial, IMG_SIZE)
    
    history = model.fit(
          train_generator,
          steps_per_epoch=1881 // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=807 // batch_size,
          verbose=1
          )
    
    accuracy = np.round(history.history['val_accuracy'][-1], 4)

    return accuracy



def create_model_deeper_wider_2(trial, IMG_SIZE):
  tf.keras.backend.clear_session()
    
  activation = 'relu'
  depth = trial.suggest_int("depth", 1, 12)
  width = trial.suggest_int("width", 16, 2048)
  
  model = Sequential()
  model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
  model.add(Dense(units=width, activation='relu',name='second', input_shape=(IMG_SIZE*IMG_SIZE*3,)))
  for i in range(1,depth):
    model.add(Dense(units=width, activation='relu',name='second_'+str(i)))
  model.add(Dense(units=8, activation='softmax'))
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  
  trial.set_user_attr("parameters", model.count_params())
  # print(model.summary())
  
  return model

def objective_2(trial):
    IMG_SIZE = 32
  
    batch_size = 8
    epochs = 20
    DATASET_DIR = 'MIT_split'
  
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            DATASET_DIR+'/train', 
            target_size=(IMG_SIZE, IMG_SIZE), 
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')  

    validation_generator = test_datagen.flow_from_directory(
            DATASET_DIR+'/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')
  
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()
    
    
    # Generate our trial model.
    model = create_model_deeper_wider_2(trial, IMG_SIZE)
    
    history = model.fit(
          train_generator,
          steps_per_epoch=1881 // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=807 // batch_size,
          verbose=1
          )
    
    accuracy = np.round(history.history['val_accuracy'][-1], 4)

    return accuracy


def experiment_going_deeper_wider(IMG_SIZE=32):
  batch_size = 32
  n_trials = 10
  
  study_name = "example-study-wide-deeper"  # Unique identifier of the study.
  storage_name = "sqlite:///{}.db".format(study_name)
  study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                              direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
  study.optimize(objective_2, n_trials=n_trials)
  
  # fig = plot_optimization_history(study)
  # fig.show()
  outputs_dir = 'outputs/'
  
  fig = plot_contour(study, params=['width','depth'])
  fig.write_image(outputs_dir+"contour_width_depth.jpeg")
  
  accuracies = []
  parameters = []
  
  for i in range(len(study.trials)):
    accuracies.append(study.trials[i].values[0])
    parameters.append(study.trials[i].user_attrs['parameters'])
  
  plt.title('Effect of increasing the number of parameters of the model')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of parameters')
  plt.scatter(parameters, accuracies)
  plt.savefig(outputs_dir+'n_parameters.jpg')
  plt.close()

def experiment_going_all(IMG_SIZE=32):
  batch_size = 32
  n_trials = 2
  
  
  study_name = "example-study"  # Unique identifier of the study.
  storage_name = "sqlite:///{}.db".format(study_name)
  study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                              direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
  study.optimize(objective, n_trials=n_trials)
  
  # fig = plot_optimization_history(study)
  # fig.show()
  outputs_dir = 'outputs/'
  
  fig = plot_param_importances(study)
  fig.write_image(outputs_dir+"param_importances.jpeg")
  
  fig = plot_contour(study, params=['width','depth'])
  fig.write_image(outputs_dir+"contour_width_depth.jpeg")
  
  fig = optuna.visualization.plot_parallel_coordinate(study)
  fig.write_image(outputs_dir+"parallel_coordinate.jpeg")
  
  accuracies = []
  parameters = []
  
  for i in range(len(study.trials)):
    accuracies.append(study.trials[i].values[0])
    parameters.append(study.trials[i].user_attrs['parameters'])
  
  plt.title('Effect of increasing the number of parameters of the model')
  plt.ylabel('Accuracy')
  plt.xlabel('Number of parameters')
  plt.plot(parameters, accuracies)
  plt.savefig(outputs_dir+'n_parameters.jpg')
  plt.close()
  
  
  
def experiment_gpu():
  im_sizes = [8, 16, 32, 64, 128]
  # im_sizes = [8, 16]
  accuracies = []
  
  for IMG_SIZE in im_sizes:
    epochs = 20
    run = wandb.init(project='M3 - Week 3 - MLP',
                    name = str(IMG_SIZE),
                    config = {
                    'learning_rate': 0.001,
                    'epochs': epochs,
                    'hidden': 2048,
                    'batch_size': 8,
                    'loss_function': 'categorical_crossentropy',
                    'optimizer': 'sgd',
                    'architecture': 'MLP',
                    'dataset': 'MIT',
                    'classes': 8
                    })
    config = wandb.config
    model, optimizer = make(config, IMG_SIZE)

    batch_size = config.batch_size

    DATASET_DIR = 'MIT_split'

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            DATASET_DIR+'/train', 
            target_size=(IMG_SIZE, IMG_SIZE), 
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')  

    validation_generator = test_datagen.flow_from_directory(
            DATASET_DIR+'/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')

    history = model.fit(
            train_generator,
            steps_per_epoch=1881 // batch_size,
            epochs=config.epochs,
            validation_data=validation_generator,
            validation_steps=807 // batch_size,
            verbose=1, callbacks=[WandbCallback()])
    
    accuracy = np.round(history.history['val_accuracy'][-1], 4)
    accuracies.append(accuracy)
    run.finish()
  
  
  plt.title('Effect of increasing the size of the images')
  plt.ylabel('Accuracy')
  plt.xlabel('Image size')
  plt.plot(im_sizes, accuracies)
  plt.savefig('outputs/im_sizes.jpg')
  plt.close()