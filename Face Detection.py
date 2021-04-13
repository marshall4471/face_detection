#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import sys


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


img = cv2.imread('jpg')


# In[5]:


plt.imshow(img)


# In[6]:


imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(imgc, cmap='gray')


# In[7]:


haar_cascade= cv2.CascadeClassifier('C://Users/PC/Downloads/haarcascade_frontalface_default.xml') 


# In[13]:


faces_rects = haar_cascade.detectMultiScale(imgc, scaleFactor = 1.1, minNeighbors = 12);
print('How many demented faces: ', len(faces_rects))


# In[9]:


for (x, y, w, h) in faces_rects:
    cv2.rectangle(imgc, (x,y), (x+w, y+h), (255, 0, 0, 3))


# In[10]:


plt.imshow(imgc)


# In[ ]:




