#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


# View example digit
index = np.random.randint(0,60000)
plt.imshow(x_train[index])
print(y_train[index])


# Given a 28 x 28 image of a hand written digit write a program that recognizes the digit.
# 
# Challenge 1: Don't use ML
# 
# Challenge 2: Use an ML algorithm other than NN
# 
# Challenge 3: Use a fully-connected NN
# 
# Challenge 4: Use a CNN

# ## Challenge 1: Don't use ML
# 
# To make things simpler, build a binary classifier. For a given image, the classifier should tell if the digit is "1" or not.

# In[4]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[5]:


# View example 1
index = np.random.randint(0,x_train[y_train==1].shape[0])
plt.imshow(x_train[y_train==1][index])
plt.colorbar()


# In[6]:


# View example not-1
index = np.random.randint(0,x_train[y_train!=1].shape[0])
plt.imshow(x_train[y_train!=1][index])
plt.colorbar()
print(y_train[y_train!=1][index])


# In[7]:


# Average of all '1' images
plt.imshow(np.mean(x_train[y_train==1], axis=0))


# In[8]:


# we can select a region of interest (ROI)
# that we think is useful for classifying images
# For example [5:20,13:16]


# In[9]:


# ROI on Average of all '1' images
plt.imshow(np.mean(x_train[y_train==1], axis=0)[5:20,13:16])
plt.colorbar()
# Mean of the ROI
np.mean(np.mean(x_train[y_train==1], axis=0)[5:20,13:16])


# In[10]:


# For a given image
# If the ROI mean pixel intensity>150
# classify it as a "1"
# else "not 1".

def naive_clf(image):
    roi = image[5:20,13:16]
    if np.mean(roi)>150:
        return 1
    else:
        return 0


# In[11]:


# Test the classifier on test images

tp = 0 # true pos
fp = 0 # false pos
for i in range(1135):
    # This should return 1
    result1 = naive_clf(x_test[y_test==1][i])
    tp += result1
    
    # This should return 0
    result2 = naive_clf(x_test[y_test!=1][i])
    fp += result2
    
tn = 1135-fp
acc = (tn+tp)/(1135*2)
print("Accuracy:", np.round(acc,2))
# Random guess can get 0.5 accuracy


# ## Challenge 2
# 
# Use an ML algorithm other than NNs.

# In[12]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)


# In[15]:


# Training
model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[16]:


predictions = model.predict(x_test)
acc = np.sum(predictions==y_test)/y_test.shape[0]
print("Accuracy:", np.round(acc,2))


# ## Challenge 3
# 
# Use a FC NN.

# In[17]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[18]:


from keras.models import Sequential
from keras.layers import Dense, Activation


# In[19]:


# Build the architecture
model = Sequential()
model.add(Dense(20, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


# In[20]:


# Set the optimizer and the loss
from keras.optimizers import SGD
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[21]:


# Convert labels to one-vs-all
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)


# In[22]:


# Flatten
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)


# In[23]:


# Train and test
H = model.fit(x_train,y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test))


# ## Challenge 4
# 
# Use a CNN.

# In[24]:


# Download the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[25]:


from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Activation, Dense


# In[26]:


# Build the architecture
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


# In[27]:


# Convert labels to one-vs-all
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)


# In[28]:


# Set the optimizer and the loss
from keras.optimizers import Adam, SGD
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# In[29]:


# Put it into suitable shape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


# In[30]:


# Train and test
H = model.fit(x_train,y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test))

