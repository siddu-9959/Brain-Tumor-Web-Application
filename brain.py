import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)

test_datagen = image.ImageDataGenerator(rescale=1./255)
train_dataset = train_datagen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
train_dataset.class_indices
test_dataset = test_datagen.flow_from_directory(
    test_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
hist = model.fit(
    train_dataset,
    epochs = 30,
    validation_data = test_dataset
)
model.save('model.h5')
from tensorflow.keras.preprocessing.image import load_img
img=image.load_img('C:/Users/srilu/Documents/GitHub/Brain-Tumor-Detection/Test/no/3 no.jpg',target_size=(224,224))
x=image.img_to_array(img)
x=x/255
x=np.expand_dims(x,axis=0)
pred=model.predict(x)
pred[0][0]
if pred[0][0] < 0.5:
    print('No tumor detected')
else:
    print('Tumor Detected')
model.evaluate_generator(test_dataset)[1]
history=hist.history.keys()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'],color='red')
plt.title('Accuracy Curves')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'],color='red')
plt.title('Loss Curves')