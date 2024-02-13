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
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.cm as cm
from wandb.keras import WandbCallback
from tensorflow.keras.layers import BatchNormalization


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


### VGG16 ###
# FROM https://keras.io/examples/vision/grad_cam/
def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
  
def vgg16_gradcam(model, last_conv_layer_name, img_array, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purposes, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    return superimposed_img


def vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing=0):
  train_set = 'MIT_small_train_1'
  root_dir = './' + train_set

  train_data_dir= root_dir + '/train'
  val_data_dir= root_dir + '/test'
  test_data_dir= root_dir + '/test'
  img_width = 224
  img_height=224
  batch_size=32
  validation_samples=807
  
  # create the base pre-trained model
  base_model = VGG16(input_shape=[img_width, img_height, 3], weights='imagenet', include_top=False)
  # plot_model(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

  x = Flatten()(base_model.output)
  x = Dense(2048)(x)
  if batchnorm == True:
    x = BatchNormalization()(x)
  x = tf.keras.layers.ReLU()(x)
  if dropout == True:
    x = tf.keras.layers.Dropout(.4)(x)   
        
  prediction = Dense(8, name='predictions', activation='softmax')(x)

  model = Model(inputs=base_model.input, outputs=prediction)
  
  for layer in base_model.layers:
      layer.trainable = False
    
      
  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),optimizer='adam', metrics=['accuracy'])
  for layer in model.layers:
      print(layer.name, layer.trainable)
  
  fig_name_prev = ''
  if dropout == True:
    fig_name_prev += '_with_dropout_'
  if batchnorm == True:
    fig_name_prev += '_with_batchnorm_'
  
  tf.keras.utils.plot_model(model, './outputs/model_vgg16'+fig_name_prev+'.png')
  
  train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    validation_split=0.2,
    horizontal_flip = True
  )

  test_datagen = ImageDataGenerator(rescale = 1./255)


  train_generator = train_datagen.flow_from_directory(train_data_dir,
          target_size=(img_width, img_height),
          batch_size=batch_size,
          class_mode='categorical')

  validation_generator = test_datagen.flow_from_directory(val_data_dir,
          target_size=(img_width, img_height),
          batch_size=batch_size,
          class_mode='categorical')
  
  
  history=model.fit(train_generator,
        steps_per_epoch=(int(400//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=validation_generator,
        validation_steps= (int(validation_samples//batch_size)+1), 
        callbacks=[],
        verbose=1)

  
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('./outputs/vgg16'+fig_name_prev+'accuracy.jpg')
  plt.close()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('./outputs/vgg16'+fig_name_prev+'loss.jpg')
  
  val_accuracy = history.history['val_accuracy'][-1] * 100
  val_loss = history.history['val_loss'][-1]
  train_accuracy = history.history['accuracy'][-1] * 100
  train_loss = history.history['loss'][-1]
  
  return model, train_generator.class_indices, val_accuracy, val_loss, train_accuracy, train_loss
  
### VGG16 ###



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
  
  train_features = np.load('train_features.npy')
  train_features = train_features.reshape((train_features.shape[0], -1))
  
  model = Sequential()
  model.add(Dense(units=width, activation='relu',name='second', input_shape=(train_features.shape[1],)))
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
  
def objective_3(trial):
    IMG_SIZE = 32
  
    batch_size = 512
    epochs = 150
  
    train_features = np.load('train_features.npy')
    test_features = np.load('test_features.npy')

    train_labels = np.load('train_labels_MIT_small_train_1.npy')
    test_labels = np.load('test_labels_MIT_small_train_1.npy')
    
    train_features = train_features.reshape((train_features.shape[0], -1))
    test_features = test_features.reshape((test_features.shape[0], -1))
    
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()
    
    # Generate our trial model.
    model = create_model_deeper_wider_2(trial, IMG_SIZE)
    
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_features, test_labels), verbose=0)
    
    accuracy = np.round(history.history['val_accuracy'][-1], 4)

    return accuracy
  

def create_optuna_vgg16_optimizer(trial):
  tf.keras.backend.clear_session()
  
  optimizer = trial.suggest_categorical('optimizer', ['SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'])
  momentum = trial.suggest_float("momentum", 0, 1)
  learning_rate = trial.suggest_float("learning_rate", 0.0001, 1)
  
  train_features = np.load('train_features.npy')
  train_features = train_features.reshape((train_features.shape[0], -1))
  
  if optimizer == 'SGD':
    opt = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate, momentum=momentum)
  elif optimizer == 'RMSprop':
    opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate, momentum=momentum)
  elif optimizer == 'Adam':
    opt = tf.keras.optimizers.experimental.Adam(learning_rate=learning_rate, ema_momentum=momentum)
  elif optimizer == 'Adagrad':
    opt = tf.keras.optimizers.experimental.Adagrad(learning_rate=learning_rate, ema_momentum=momentum)
  elif optimizer == 'Adadelta':
    opt = tf.keras.optimizers.experimental.Adadelta(learning_rate=learning_rate, ema_momentum=momentum)
  elif optimizer == 'Adamax':
    opt = tf.keras.optimizers.experimental.Adamax(learning_rate=learning_rate, ema_momentum=momentum)
  elif optimizer == 'Nadam':
    opt = tf.keras.optimizers.experimental.Nadam(learning_rate=learning_rate, ema_momentum=momentum)
  
  model = Sequential()
  model.add(Dense(units=2048, activation='relu',name='second', input_shape=(train_features.shape[1],)))
  model.add(Dense(units=8, activation='softmax'))
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  
  trial.set_user_attr("parameters", model.count_params())
  
  return model

def objective_optimizer(trial):
    batch_size = 2048
    epochs = 150
  
    train_features = np.load('train_features.npy')
    test_features = np.load('test_features.npy')

    train_labels = np.load('train_labels_MIT_small_train_1.npy')
    test_labels = np.load('test_labels_MIT_small_train_1.npy')
    
    train_features = train_features.reshape((train_features.shape[0], -1))
    test_features = test_features.reshape((test_features.shape[0], -1))
    
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()
    
    
    # Generate our trial model.
    model = create_optuna_vgg16_optimizer(trial)
    
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_features, test_labels), verbose=0)
    
    accuracy = np.round(history.history['val_accuracy'][-1], 4)

    return accuracy

def experiment_optimizer():
  n_trials = 2
  
  study_name = "vgg16-study-optimizers"  # Unique identifier of the study.
  storage_name = "sqlite:///{}.db".format(study_name)
  study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                              direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
  study.optimize(objective_optimizer, n_trials=n_trials)
  
  outputs_dir = 'outputs/'
  
  fig = plot_contour(study, params=['optimizer','learning_rate'])
  fig.write_image(outputs_dir+"contour_optimizer.jpeg")
  

def experiment_going_deeper_wider(IMG_SIZE=32):
  n_trials = 2
  
  study_name = "vgg16-study-wide-deeper"  # Unique identifier of the study.
  storage_name = "sqlite:///{}.db".format(study_name)
  study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                              direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
  study.optimize(objective_3, n_trials=n_trials)
  
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