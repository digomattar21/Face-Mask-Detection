from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import os
from PIL import Image

#Initial learning rate
INIT_LR = 1e-4
#20 epochs
EPOCHS = 20
#Back size 
BS = 32

directory = r'dataset'
categories = ['with_mask' , 'without_mask']
print('[INFO] loading images')

data=[]
labels =[]


#opens with_mask and withou_mask foklders, goes thru the images and then grabs the path of all of them and appends innto the labels and data
#Also converts the image to 224 224 and converts to array
for category in categories:
    path = os.path.join(directory,category)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)
    


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#Converts data to array 
data=np.array(data, dtype='float32')
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, stratify=labels)


#Data augmentation (constructor), bascally it generates images from 1 image to optmize how a picture can be presented
aug = ImageDataGenerator(
    rotation_range =20,
    zoom_range = 0.15,
    width_shift_range=0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip=True
)

#Load the mobilenetv2 
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

headModel= baseModel.output
headModel= AveragePooling2D(pool_size=(7,7))(headModel)
headModel= Flatten(name='flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel= Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model= Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
    layer.treinable = False

print('[INFO] compiling model...')

opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer =opt, metrics=['accuracy'])

#train head of network
print('[INFO] trainig head of network...')
H=model.fit(
    aug.flow(trainX,trainY, batch_size =BS),
    steps_per_epoch= len(trainX)//BS,
    validation_data= (testX,testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS)

print('[INFO] evaluating model...')
predIdxs = model.predict(testX,batch_size=BS)

predIdxs= np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1),predIdxs,target_names=lb.classes_))

print('[INFO] saving mask detector model')
model.save('mask_detector.model', save_format='h5')

N= EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0,N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0,N), H.history['accuracy'], label='val_acc')
plt.plot(np.arange(0,N), H.history['val_accuracy'], label='val_acc')
plt.title(' Training Loss and Accuracy')
plt.xlabel(' Epoch #')
plt.ylabel(' Loss/Accuracy')
plt.legend(loc='lower_left')
plt.savefig('plot.png')


