from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from utils import *
from utils_data_augm import * 
from sklearn.svm import SVC
from keras.models import *
from keras.layers import *
from sklearn.manifold import TSNE
import seaborn as sns 
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL 
import shutil
import glob

def histogramIntersection(M, N):
  K = np.zeros((M.shape[0], N.shape[0]), dtype=float)

  for j, _ in enumerate(N):
    for i, _ in enumerate(M):
      K[i][j] = np.sum(np.minimum(M[i], N[j]))

  return K



train_features = np.load('train_features.npy')
test_features = np.load('test_features.npy')

train_labels = np.load('train_labels_MIT_small_train_1.npy')
test_labels = np.load('test_labels_MIT_small_train_1.npy')


train_features = train_features.reshape((train_features.shape[0], -1))
test_features = test_features.reshape((test_features.shape[0], -1))


### START VGG16 layers modifications ###
val_accuracies, val_losses, train_accuracies, train_losses = [], [], [], []
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing=0)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.1)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.2)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.3)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing=0.4)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.5)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.6)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.7)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.8)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 0.9)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False, label_smoothing = 1)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)

methods = ['No label smoothing(0)', 'Label smoothing(0.1)', 'Label smoothing(0.2)', 'Label smoothing(0.3)',
           'No label smoothing(0.4)', 'Label smoothing(0.5)', 'Label smoothing(0.6)', 'Label smoothing(0.7)',
           'No label smoothing(0.8)', 'Label smoothing(0.9)', 'Label smoothing(1)']
plt.figure(figsize=(12,8))
X_axis = np.arange(len(methods))
plt.bar(X_axis - 0.4, train_losses, 0.2, label='Train loss')
plt.bar(X_axis - 0.2, val_losses, 0.2, label='Validation loss')
plt.bar(X_axis, train_accuracies, 0.2, label='Train accuracy')
plt.bar(X_axis + 0.2, val_accuracies, 0.2, label='Validation accuracy')

plt.yticks(np.arange(0, 110, 10))
plt.xticks(X_axis, methods, rotation=45, ha="right")
plt.xlabel('Classification method')
plt.ylabel('Performance')
plt.title('Performance of our VGG16 extracted features \n with our MLP modified classifier')
plt.legend()
plt.ylim(0,100)

plt.tight_layout()
plt.savefig('./outputs/label_smoothing.jpg')
plt.close()
### END VGG16 layers modifications ###

### START VGG16 layers modifications ###
val_accuracies, val_losses, train_accuracies, train_losses = [], [], [], []
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=False)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=True, batchnorm=False)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=False, batchnorm=True)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2, dropout=True, batchnorm=True)
val_accuracies.append(val_accuracy)
val_losses.append(val_loss)
train_accuracies.append(train_accuracy)
train_losses.append(train_loss)

methods = ['No modification', 'Dropout(0.4)', 'Batchnorm', 'Dropout(0.4) + Batchnorm']
plt.figure(figsize=(12,8))
X_axis = np.arange(len(methods))
plt.bar(X_axis - 0.4, train_losses, 0.2, label='Train loss')
plt.bar(X_axis - 0.2, val_losses, 0.2, label='Validation loss')
plt.bar(X_axis, train_accuracies, 0.2, label='Train accuracy')
plt.bar(X_axis + 0.2, val_accuracies, 0.2, label='Validation accuracy')

plt.xticks(X_axis, methods)
plt.xlabel('Classification method')
plt.ylabel('Performance')
plt.title('Performance of our VGG16 extracted features \n with our MLP modified classifier')
plt.legend()
plt.ylim(0,100)


plt.savefig('./outputs/layer_modification.jpg')
plt.close()
### END VGG16 layers modifications ###

### START VGG16 gradcam ###
model, indices, val_accuracy, val_loss, train_accuracy, train_loss = vgg16(number_of_epoch=2)
losses = {}
stop = 0
top_preds = 1

for class_path in glob.glob('./MIT_small_train_1/test/*'):
  im_class = os.path.basename(class_path)
  for im_path in glob.glob(class_path + '/*'):
    im_name = os.path.basename(im_path)
    
    img_array = preprocess_input(get_img_array(im_path, size=(224,224,3)))
    
    model.layers[-1].activation = None
    pred = model.predict(img_array)[0]
    softmax = tf.keras.layers.Softmax()
    pred_softmax = softmax([pred]).numpy()[0]
    
    indices = list(indices)
    results = []
    
    top_indices = pred.argsort()[-top_preds:][::-1]
    
    # If we wanted to take all results    
    # result = [indices[i] + ' ' + str(pred[i]) + ' ' + str(pred_softmax[i]) for i in top_indices]
    
    # In this case we just want the top 1!
    index = top_indices[0]
    losses[pred[index]] = {
      'prediction': indices[index],
      'groundtruth': im_class,
      'loss': pred[index],
      'softmax': pred_softmax[index],
      'name': im_name,
    }
    
    
    if stop > 50000:
      break
    stop+=1
    
# Items with less confidence
losses = dict(sorted(losses.items(), reverse=True))
max_loss_items = 20
plt.figure(figsize=(20, 12))
plt.suptitle('Top 20 images where the model is less confident by loss\n The titles of each image are in the form groundtruth\\prediction\\loss',
             fontsize=22, weight='bold')
i = 1
for key, value in losses.items(): 
  im_path = './MIT_small_train_1/test/'+value['groundtruth']+'/'+value['name']
  plt.subplot(4, 5, i)
  im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
  plt.axis('off')
  plt.title(value['groundtruth'] + '\\' + value['prediction'] + '\\' + str(value['loss']))
  plt.imshow(im)
  
  img_array = preprocess_input(get_img_array(im_path, size=(224,224,3)))
  heatmap = vgg16_gradcam(model, 'block5_conv3', img_array)
  shutil.copyfile(im_path, './outputs/gradcam/'+value['name'])
  save_and_display_gradcam(im_path, heatmap, cam_path='./outputs/gradcam/'+value['name'].split('.')[0]+'_cam.jpg')
  i+=1
  if i > max_loss_items:
    break
plt.tight_layout()
plt.savefig('./outputs/most_pred_loss.jpg')

# Items with most confidence
losses = dict(sorted(losses.items(), reverse=False))
max_loss_items = 20
plt.figure(figsize=(20, 12))
plt.suptitle('Top 20 images where the model is most confident by loss\n The titles of each image are in the form groundtruth\\prediction\\loss',
             fontsize=22, weight='bold')
i = 1
for key, value in losses.items(): 
  im_path = './MIT_small_train_1/test/'+value['groundtruth']+'/'+value['name']
  plt.subplot(4, 5, i)
  im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
  plt.axis('off')
  plt.title(value['groundtruth'] + '\\' + value['prediction'] + '\\' + str(value['loss']))
  plt.imshow(im)
  
  img_array = preprocess_input(get_img_array(im_path, size=(224,224,3)))
  heatmap = vgg16_gradcam(model, 'block5_conv3', img_array)
  shutil.copyfile(im_path, './outputs/gradcam/'+value['name'])
  save_and_display_gradcam(im_path, heatmap, cam_path='./outputs/gradcam/'+value['name'].split('.')[0]+'_cam.jpg')
  i+=1
  if i > max_loss_items:
    break
plt.tight_layout()
plt.savefig('./outputs/less_pred_loss.jpg')

im_class = 'Opencountry'
im_name = 'land625'
im_path = './MIT_small_train_1/train/'+im_class+'/'+im_name+'.jpg'
heatmap = vgg16_gradcam(model, 'block5_conv3', img_array)
shutil.copyfile(im_path, './outputs/gradcam/'+im_name+'.jpg')
plt.figure(figsize=(5,5), dpi=150)
plt.imshow(heatmap)
plt.tight_layout()
plt.savefig('./outputs/gradcam/'+im_name+'_heatmap.jpg')
plt.close()
save_and_display_gradcam(im_path, heatmap, cam_path='./outputs/gradcam/'+im_name+'_cam.jpg')
### END VGG16 gradcam ###




### Test optimizer ###
experiment_optimizer()


### Test wide and deep ###
experiment_going_deeper_wider()

### Test data augmentation ###
experiment_data_augmentation()


### TSNE ###
tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(train_features, np.argmax(train_labels, axis=1))
palette = sns.hls_palette(8)
sns.set(rc={'figure.figsize':(12,9)})
s = sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = np.argmax(train_labels, axis=1), palette=palette, legend='full')
s.set_title('t-SNE of the VGG16 last convolutional layer output',size=20)
plt.legend(['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding'])
plt.tight_layout()
plt.savefig('./outputs/tsne_vgg16.jpg')

### NORMAL VGG16 ###

model = Sequential()

model.add(Dense(units=2048, activation='relu',name='second', input_shape=(train_features.shape[1],)))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

epochs = 2
history = model.fit(x=train_features, y=train_labels, batch_size=32, epochs=epochs, validation_data=(test_features, test_labels))

print(history.history['val_accuracy'][-1])

plt.figure(figsize=(5,5), dpi=150)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy of frozen VGG16 output\n as MLP input descriptor')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,epochs+1,20))
plt.tight_layout()
plt.legend()
plt.savefig('./outputs/accuracy_vgg16_mlp.jpg')
plt.close()

plt.figure(figsize=(5,5), dpi=150)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss of frozen VGG16 output\n as MLP input descriptor')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0,epochs+1,20))
plt.legend()
plt.tight_layout()
plt.savefig('./outputs/loss_vgg16_mlp.jpg')
plt.close()

### END normal VGG16


### START ACCURACIES VGG16

res = model.predict(test_features)
res2 = np.argmax(res, axis=1)

accuracies = []
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title('Accuracies of the different classes with VGG16')
label=['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']

for i in range(0,8):
  res3=np.where(res2==i)
  res4=np.argmax(test_labels, axis=1)
  res5=np.where(res4==i)
  accuracy = accuracy_score(res2[res3],res4[res3])
  accuracies.append(accuracy)
  ax.bar(i, accuracies, label=label[i])

ax.set_ylabel('Accuracy')
ax.set_xlabel('Classes')
ax.set_xticks(np.arange(0,8))
ax.set_xticklabels(label, rotation=45)
ax.set_ylim(0.7,1)
plt.tight_layout()
fig.savefig('./outputs/accuracies.jpg')

### END ACCURACIES VGG16

### Start accuracies SVM
accuracies_mlp_svm = []
clfs_names = ["SVM - Linear kernel", "SVM - Poly kernel", "SVM - rbf kernel", "SVM - Hist. Inters. Kernel", "SVM - Sigmoid kernel"]
clfs = [SVC(kernel='linear',gamma='auto'), SVC(kernel='poly',gamma='auto'), SVC(kernel='rbf',gamma='auto'), 
        SVC(kernel=histogramIntersection,gamma='auto'), SVC(kernel='sigmoid',gamma='auto')]
for clf in clfs:
  clf.fit(train_features, np.argmax(train_labels, axis=1))
  accuracy = clf.score(test_features, np.argmax(test_labels, axis=1))
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
plt.savefig('outputs/cnn_svm_accuracy.jpg')
plt.close()

print(clfs_names)
print(accuracies_mlp_svm)

### End accuracies SVM