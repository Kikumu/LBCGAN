#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image 
from sklearn import preprocessing
import numpy as np


# In[2]:


class rescalor_class():
    def __init__(self, min_, max_):
        self.min_scale = min_
        self.max_scale = max_
    
    def rescale_rgb_channels(self, image, normalize=False):
        for idx_, image_channel in enumerate(image):
            rescale_func = preprocessing.MinMaxScaler(feature_range=(self.min_scale, self.max_scale))
            rescaled_channel = rescale_func.fit_transform(np.squeeze(image_channel))
            
            standard_scaler_func = preprocessing.StandardScaler().fit(rescaled_channel)
            rescaled_channel = standard_scaler_func.transform(rescaled_channel)

            rescaled_channel = np.expand_dims(rescaled_channel, axis=0)
            if idx_ == 0:
                stacked_channel=rescaled_channel
            else:
                stacked_channel = np.append(stacked_channel, rescaled_channel, axis=0)
        return stacked_channel
    
    def rescale_image(self, image):
        for i, image_batch in enumerate(image):
            rescaled_batch = self.rescale_rgb_channels(image_batch)
            rescaled_batch = np.expand_dims(rescaled_batch, axis=0)
            if i == 0:
                stacked_batch = rescaled_batch
            else:
                stacked_batch = np.append(stacked_batch, rescaled_batch, axis=0)
        return stacked_batch


# In[3]:


class ThumbnailDataloader(Dataset):
    def __init__(self, dataframe, transform=None, resize=(126, 126)):
        self.annotations = dataframe
        self.transform = transform
        self.resize = resize
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        #print(idx)
        thumbnail_image = Image.open(self.annotations.iloc[idx].values[0])
        thumbnail_image = thumbnail_image.resize((self.resize[0], 
                                                  self.resize[1]), Image.ANTIALIAS)
        if self.transform:
            thumbnail_image = self.transform(thumbnail_image)
        return thumbnail_image
    
#no need for labels cause images from thumbnail will always be 1, randomized generated images will always be 0

class NumpyDataloader(Dataset):
    def __init__(self, dataframe, transform=None, resize=(126, 126)):
        self.annotations = dataframe
        self.transform = transform
        self.resize = resize
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        #print(idx)
        thumbnail_npy = np.load(self.annotations.iloc[idx].values[0])
        if self.transform:
            thumbnail_npy = self.transform(thumbnail_npy)
        return thumbnail_npy
# In[4]:


def load_filenames_array(root_dir, extension):
    thumbnail_img_path_array = []#to be saved to a dataframe for dataloading
    for idx_folder in os.listdir(root_dir):#open root folder(thumbnails)
        if 'LICENSE' not in idx_folder:#all idx folder except license text
            for idx_thumbnail in os.listdir(os.path.join(root_dir, idx_folder)):
                if idx_thumbnail.endswith(extension):
                    thumbnail_img_path_array.append(os.path.join(root_dir, idx_folder, idx_thumbnail))
    return thumbnail_img_path_array


# In[26]:


#root_dir = '../face_data/thumbnails/'
#extension = '.png'
#thumbnails_locations = load_filenames_array(root_dir, extension)
#thumbnails_locations = [location.replace('\\', '/')for location in thumbnails_locations]


# In[ ]:




