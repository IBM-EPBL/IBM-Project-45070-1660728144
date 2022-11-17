#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers


# In[2]:


# Initialing the CNN
classifier = Sequential()


# In[3]:


# Step 1 - Convolution Layer
classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))


# In[4]:


#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))


# In[5]:


# Adding second convolution layer
classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))


# In[12]:


#Step 3 - Flattening
classifier.add(Flatten())


# In[13]:


#Step 3 - Flattening
classifier.add(Flatten())


# In[14]:


#Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10, activation = 'softmax'))


# In[15]:


#Compiling The CNN
classifier.compile(
              optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


# In[16]:


#Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[17]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[20]:


training_set = train_datagen.flow_from_directory(
        r'C:\Users\DELL\Desktop\train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[21]:


test_set = test_datagen.flow_from_directory(
        r"C:\Users\DELL\Desktop\test",
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


# In[27]:


model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=40,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=70,kernel_initializer='random_uniform',activation='relu'))
model.add(Dense(units=6,kernel_initializer='random_uniform',activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
x_train = train_datagen.flow_from_directory(r"C:\Users\DELL\Desktop\fruit-dataset\fruit-dataset\train",target_size = (128,128), batch_size = 32,class_mode = 'categorical')
x_test = test_datagen.flow_from_directory(r"C:\Users\DELL\Desktop\fruit-dataset\fruit-dataset\test",target_size = (128,128), batch_size = 32,class_mode = 'categorical')
model.fit(x_train,steps_per_epoch=168,epochs=3,validation_data=x_test,validation_steps=52)


# In[28]:


#Saving the model
import h5py
classifier.save('Trained_Model.h5')

