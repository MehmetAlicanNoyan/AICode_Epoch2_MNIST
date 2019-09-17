#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


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

# In[ ]:




