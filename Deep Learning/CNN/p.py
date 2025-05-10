import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import tensorflow
from tensorflow.keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(16, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())

cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(64, activation="relu"))
cnn.add(Dense(32, activation="relu"))
cnn.add(Dense(16, activation="relu"))
cnn.add(Dense(8, activation="relu"))
cnn.add(Dense(1, activation="sigmoid"))


cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r"C:\Users\vishw\Downloads\train",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        r"C:\Users\vishw\Downloads\test",
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

history = cnn.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=5,
        validation_data=test_generator,
        )
print(history)

from keras.preprocessing import image
img = image.load_img(r"C:\Users\vishw\Downloads\test\dogs\dog_303.jpg", target_size=(64,64))
img = image.img_to_array(img)
img = img/255
img = img.reshape(1,64,64,3)
classes = cnn.predict(img)
print(classes)
if classes[0]>0.5:
    print("dog")
else:
    print("cat")
    
