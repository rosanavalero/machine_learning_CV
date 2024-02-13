import os
import getpass
from utils import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import wandb
from wandb.keras import WandbMetricsLogger
from wandb.keras import WandbCallback
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import seaborn as sns




print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



IMG_SIZE = 32

################## EXPERIMENTS WITH SOME HYPERPARAMETERS ##################

### Experiment going deeper and wider
experiment_going_deeper_wider(IMG_SIZE)

### Experiment on every hyperparameter
experiment_going_all(IMG_SIZE)


# sys.exit()

### Experiment GPU
wandb.login()
experiment_gpu()
# sys.exit()

################## MLP MODEL BASELINE ##################
load_model = True
MODEL_FNAME = 'my_first_mlp.h5'

        
if load_model == False:
        print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')
        
        run = wandb.init(project='M3 - Week 3 - MLP',
                        config = {
                        'learning_rate': 0.001,
                        'epochs': 150,
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
        
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylim(0,1)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('outputs/mlp_accuracy.jpg')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('outputs/mlp_loss.jpg')
        
        model.save(MODEL_FNAME)
else:
        model = tf.keras.models.load_model(MODEL_FNAME)

model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

################## USING PRETRAINED FEATURES ON SVM CLASSIFIER ##################
X, train_images, test_images, train_labels, test_labels = get_files(IMG_SIZE)

train_images_scaled, test_images_scaled = normalize_data(train_images, test_images, IMG_SIZE, normalization='unit')
Train_images_scaled = model_layer.predict(train_images_scaled)
Test_images_scaled = model_layer.predict(test_images_scaled)

scaler = StandardScaler()
Train_images_scaled = scaler.fit_transform(Train_images_scaled)
Test_images_scaled = scaler.fit_transform(Test_images_scaled)
train_labels_orig = pickle.load(open('train_labels.dat','rb'))
test_labels_orig = pickle.load(open('test_labels.dat','rb'))

accuracies_mlp_svm = []
clfs_names = ["SVM - Linear kernel", "SVM - Poly kernel", "SVM - rbf kernel", "SVM - Hist. Inters. Kernel", "SVM - Sigmoid kernel"]
clfs = [SVC(kernel='linear',gamma='auto'), SVC(kernel='poly',gamma='auto'), SVC(kernel='rbf',gamma='auto'), 
        SVC(kernel=histogramIntersection,gamma='auto'), SVC(kernel='sigmoid',gamma='auto')]
for clf in clfs:
  clf.fit(Train_images_scaled, train_labels_orig)
  accuracy = clf.score(Test_images_scaled, test_labels_orig)
  print('Accuracy: ' + str(round(accuracy, 3)))
  accuracies_mlp_svm.append(accuracy)
plt.figure()
plt.plot(np.arange(len(clfs_names)), accuracies_mlp_svm)
plt.title("Accuracy of the classification")
plt.xlabel("Classifier model")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xticks(np.arange(len(clfs_names)), labels=clfs_names, rotation=45, ha="right")
plt.tight_layout()
plt.savefig('outputs/mlp_svm_accuracy.jpg')
plt.close()

tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(Train_images_scaled, train_labels_orig)
palette = sns.hls_palette(8)
sns.set(rc={'figure.figsize':(12,9)})
s = sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = train_labels_orig, palette=palette, legend='full')
# s = sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1])
s.set_title('t-SNE of the MLP hidden layer output (SVM input)',size=20)
fig = s.get_figure()
fig.savefig('outputs/tsne.jpeg')

################## EXPERIMENTS WITH PATCH-BASED MLP MODEL AS A FEATURE EXTRACTOR ##################
PATCH_SIZE  = 64
MODEL_FNAME = 'patch_based_mlp.h5'
PATCHES_DIR = 'data/MIT_split_patches'+str(PATCH_SIZE)
DATASET_DIR = 'MIT_split'
load_model = True
epochs = 150

run = wandb.init(project='M3 - Week 3 - MLP',
                        config = {
                        'learning_rate': 0.001,
                        'epochs': epochs,
                        'hidden': 2048,
                        'batch_size': 8,
                        'loss_function': 'categorical_crossentropy',
                        'optimizer': 'sgd',
                        'architecture': 'MLP patch',
                        'dataset': 'MIT',
                        'classes': 8
                        })

config = wandb.config
batch_size = config.batch_size
        
if load_model == False:
        model, optimizer = make(config, PATCH_SIZE)
        
        
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                PATCHES_DIR+'/train',
                target_size=(PATCH_SIZE, PATCH_SIZE),
                batch_size=batch_size,
                classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
                class_mode='categorical')  

        validation_generator = test_datagen.flow_from_directory(
                PATCHES_DIR+'/test',
                target_size=(PATCH_SIZE, PATCH_SIZE),
                batch_size=batch_size,
                classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
                class_mode='categorical')

        history = model.fit(
                train_generator,
                # steps_per_epoch=18810 // batch_size,
                epochs=epochs,
                validation_data=validation_generator,)
                # validation_steps=8070 // batch_size)

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('MLP-patch based accuracy')
        plt.ylim(0,1)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('outputs/mlp_patch_based_accuracy.jpg')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('MLP-path based loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('outputs/mlp_patch_based_loss.jpg')
        
        model.save_weights(MODEL_FNAME)

model, optimizer = make(config, PATCH_SIZE, phase='test')
model.load_weights(MODEL_FNAME)


accuracies_all = []
methods = ['mean', 'max', 'min', 'relu', 'linear', 'sigmoid']


def patch_based_evaluate(method='mean'):
        directory = DATASET_DIR+'/test'
        classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
        correct = 0.
        total   = 807
        count   = 0

        accuracies = []
        for class_dir in os.listdir(directory):
                cls = classes[class_dir]
                for imname in os.listdir(os.path.join(directory,class_dir)):
                        im = Image.open(os.path.join(directory,class_dir,imname))
                        patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=1)
                        out = model.predict(patches/255.)
                        if method == 'mean':
                                predicted_cls = np.argmax(softmax(np.mean(out,axis=0)))
                        elif method == 'max':
                                predicted_cls = np.argmax(softmax(np.max(out,axis=0)))
                        elif method == 'min':
                                predicted_cls = np.argmax(softmax(np.min(out,axis=0)))
                        elif method == 'relu':
                                predicted_cls = np.argmax( np.maximum(np.mean(out, axis=0), 0) )
                        elif method == 'linear':
                                predicted_cls = np.argmax( np.mean(out, axis=0))
                        elif method == 'sigmoid':
                                predicted_cls = np.argmax( 1/(1+np.exp(-np.mean(out, axis=0))) )
                        if predicted_cls == cls:
                                correct+=1
                        count += 1
    
        print('Test Acc. = '+str(correct/total)+'\n')
        return correct/total

for method in methods:
        accuracy = patch_based_evaluate(method)
        accuracies_all.append(accuracy)


plt.bar(methods, accuracies_all)
plt.title('MLP patch-based with different aggregation methods')
plt.ylabel('Validation accuracy')
plt.ylim(0,1)
plt.xticks(np.arange(len(methods)), labels=methods, rotation=45, ha="right")
plt.tight_layout()
plt.savefig('outputs/mlp_patch_based_validation.jpg')