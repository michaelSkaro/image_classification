#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Python version
import sys

print('Python: {}'.format(sys.version))
# scipy
import scipy

print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy

print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib.pyplot as plt

#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas

print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn

print('sklearn: {}'.format(sklearn.__version__))

from pandas import read_csv


# In[ ]:





# In[7]:


from os import listdir
from os.path import isfile, join
import numpy
import cv2


# In[9]:


import os
os.getcwd()
os.chdir('/home/mskaro1/storage/Machine_Learning/images/')
# read in data file that will act as our files for training

import pandas as pd
from io import StringIO

data=pd.read_csv("/home/mskaro1/storage/Machine_Learning/images/class_list.txt",delimiter="\t")

import cv2
import glob
import numpy as np


# In[10]:


get_ipython().system('conda list')


# In[18]:


get_ipython().system('conda list ')


# In[20]:


import PIL
from PIL import Image


# In[29]:


# make paddings for the images

from PIL import Image

def make_square(im, min_size=1040, fill_color=(255, 255, 255, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


# In[30]:


test_image = Image.open('1000-1001.tif')
new_image = make_square(test_image)
new_image.show()
#img.save(‘resized_image.jpg')


# In[32]:


plt.imshow(new_image)




from io import StringIO

data=pd.read_csv("/home/mskaro1/storage/Machine_Learning/images/class_list.txt",delimiter="\t")



# read in the data and use the helper functions to resize the data

import cv2
import glob
import numpy as np

file_names = data.loc[ : , 'IMG' ]

for i in file_names:
    test_image = Image.open(i)
    new_image = make_square(test_image)
    new_image.save(i+'resized_image.tif')
   


from PIL import Image


for image_file_name in os.listdir('/home/mskaro1/storage/Machine_Learning/images/'):
    if image_file_name.endswith(".tif"):
        
        print("processing imaeg : " + image_file_name)
        im = Image.open("/home/mskaro1/storage/Machine_Learning/images/" + image_file_name)
        new_width  = 1040
        new_height = 1040
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(image_file_name + "_small" + ".tif")




