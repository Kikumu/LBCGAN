#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


 
def decay_gauss_std(net):
    for m in net.modules():
        if isinstance(m, Gaussian_noise):
            m.decay_step()




# In[2]:
class Gaussian_noise(nn.Module):
    def __init__(self, total_epochs=500, std_dev=0.2, min_std_dev=0, decay_rate=0.9):
        super().__init__()
        self.initial_std_dev = std_dev
        self.min_std_dev = min_std_dev
        self.decay_ratio = decay_rate
        self.total_epochs = total_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 1
        self.current_std_dev = std_dev
        self.to(self.device)
        
        
    def decay_step(self):
        if self.current_epoch < self.total_epochs:
            self.current_epoch+=1
        decay_std_dev = self.total_epochs * self.decay_ratio
        current_std_dev = 1.0 - self.current_epoch/decay_std_dev
        current_std_dev *= self.initial_std_dev - self.min_std_dev
        current_std_dev+=self.min_std_dev
        self.current_std_dev = np.clip(current_std_dev,
                                         self.min_std_dev,
                                         self.initial_std_dev)
    
    def forward(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
        random_noise = torch.empty_like(data).normal_(std=self.current_std_dev)
        return data + random_noise.to(self.device)

# In[5]:
class discriminator_conv(nn.Module):
    def __init__(self,
                 total_epochs,
                 label_conditioning_linear_size,
                 label_vector_size,
                 embedding_vector_size,
                 hidden_channels = (512, 256, 128, 64, 32),
                 final_linear = 1,
                 stride = 2,
                 kernel_size = 2,
                 bias = False
                ):
        super(discriminator_conv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_conditioning_linear_size = label_conditioning_linear_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = 1
        self.total_epochs = total_epochs
        self.leaf = self.label_embedding_layer(label_vector_size,
                                              embedding_vector_size)
        self.tree = nn.Sequential(*[
            self.discriminator_body(hidden_channels[i], hidden_channels[i + 1]) for i in range(len(hidden_channels) - 1)])
        self.root_score = self.root(541696)
        self.to(self.device)
    
    #bchw - b,3,1,w
    def label_embedding_layer(self, 
                        label_vector_size,
                        embedding_dim_size
                       ):
        label_conditioning_layer = [nn.Embedding(label_vector_size,
                                             embedding_dim_size)]
        label_conditioning_layer.append(nn.Linear(embedding_dim_size,
                                              self.label_conditioning_linear_size))
        return nn.Sequential(*label_conditioning_layer)
    

    def discriminator_body(self, 
                          in_channels,
                          out_channels,
                          noise=True):
        if out_channels > 32:
            self.bias = False
        else:#activate bias only on initial layer
            self.bias = True
        
        if out_channels == 128:
            self.stride  = 1
        discriminator_layer = [nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        stride = self.stride,
                                        kernel_size = self.kernel_size,
                                        bias = self.bias
                                       )]
        if noise == True:
            discriminator_layer.append(Gaussian_noise(self.total_epochs))
        if out_channels!=32:#do not batchnorm initial layer
            discriminator_layer.append(nn.BatchNorm2d(out_channels))
        discriminator_layer.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*discriminator_layer)

    def root(self, flattened_tensor):
        discriminator_root_layer = [Gaussian_noise(self.total_epochs)]
        discriminator_root_layer.append(nn.Linear(flattened_tensor, 1).to(self.device))
        return nn.Sequential(*discriminator_root_layer)
    
    def forward(self, image_, conditioning_label):
        if not isinstance(image_, torch.Tensor):
            image_ = torch.tensor(image_, dtype=torch.float32).to(self.device)
        batch_size = image_.shape[0]
        conditioning_label = conditioning_label.type(torch.LongTensor).to(self.device)
        print('label shape: ', conditioning_label.shape)
        conditioning_label = self.leaf(conditioning_label)
        #BCHWCn
        print(conditioning_label.shape)
        conditioning_label = conditioning_label[:,:,-1,:]#we only want embeddinglayer data
        conditioning_label = conditioning_label.view(batch_size,
                                                    3,
                                                    128,
                                                    128)
        print(conditioning_label.shape)
        #concatenate
        print('image shape: ', image_.shape)
        print('label shape: ', conditioning_label.shape)
        image_ = torch.cat((image_, conditioning_label),1)
        print('concatenated shape: ', image_.shape)
        image_ = self.tree(image_)
        accumilated_flattening_size = image_.shape[1]*image_.shape[2]*image_.shape[3]
        image_ = torch.flatten(image_)
        image_ = image_.view(batch_size, accumilated_flattening_size)
        print(image_.shape)
        image_ = self.root_score(image_)
        return image_
    
    
    
    
    
    
    

class generator_upsample(nn.Module):
    def __init__(self,
                latent_dim_in,
                latent_dim_out,
                conditioning_dim_out_e,
                conditioning_dim_out,
                conditioning_dim_in_e,
                conditioning_dim_in, 
                hidden_channels=(512, 256, 128, 64, 32),
                kernel_size_head = 2,
                stride_head = 1,
                upsample_body_factor = 3,
                out_channel_generator=3,
                padding=1,
                bias=False):
        super(generator_upsample, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_size_head = kernel_size_head
        self.stride_head = stride_head
        self.bias = bias
        self.padding = padding
        self.upsample_body_factor = upsample_body_factor
        self.hidden_channels_head = hidden_channels[0]
        self.label_conditioning = self.label_conditioning_layer( 
                                conditioning_dim_out,
                                conditioning_dim_in,
                                conditioning_dim_in_e,
                                conditioning_dim_out_e)
        self.latent_conditioning = self.latent_linear_layer(
                                            latent_dim_in,
                                            latent_dim_out)
        self.main_head = self.head()
        self.upsample_body = (nn.Sequential(*
                                  [self.body(hidden_channels[i],
                                   hidden_channels[i + 1]) for i in range(len(hidden_channels) - 1)]
                                 ))
        self.generated_img_Layer = self.generate_layer(hidden_channels[-1], 
                                                                      out_channel_generator)
        self.to(self.device)
    
    def label_conditioning_layer(self, 
                                conditioning_dim_out,
                                conditioning_dim_in,
                                conditioning_dim_in_e,
                                conditioning_dim_out_e
                                ):
        label_layer = [nn.Embedding(conditioning_dim_in_e,
                                   conditioning_dim_out_e)]
        
        label_layer.append(nn.Linear(conditioning_dim_in,
                                    conditioning_dim_out))
        
        return nn.Sequential(*label_layer)
    
    def latent_linear_layer(self,
                           latent_dim_in,
                           latent_dim_out):
        latent_layer = [nn.Linear(latent_dim_in,
                                 latent_dim_out)]
        latent_layer.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*latent_layer)
        
    def head(self):
        head_layers = [nn.Conv2d(in_channels = 1024,
                                 out_channels = self.hidden_channels_head,
                                 kernel_size = self.kernel_size_head,
                                 stride = self.stride_head,
                                 bias = False
                                )
                      ]
        head_layers.append(nn.BatchNorm2d(self.hidden_channels_head))
        head_layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*head_layers)
    
    def body(self,
            in_channels,
            out_channels,
            dropout = True,
            dropout_rate = 0.05,
            kernel_body_size = 2,
            stride_body_size = 1,
            padding_body = 1,
            bn=True):
        upsample_body = [nn.Upsample(scale_factor=self.upsample_body_factor, mode='bilinear', align_corners=True)]
        upsample_body.append(nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_body_size,
                                       stride=stride_body_size,
                                       padding = padding_body,
                                       bias = self.bias))
        if bn == True:
            upsample_body.append(nn.BatchNorm2d(out_channels))
            
        upsample_body.append(nn.LeakyReLU(0.2))
        
        if out_channels == 32:
            dropout = False
        if dropout == True:
            upsample_body.append(nn.Dropout2d(dropout_rate))
        return nn.Sequential(*upsample_body)
        
    def generate_layer(self,
                      in_channels,
                      out_channels,
                      generator_bias =True,
                      generator_kernel_size = 2,
                      generator_stride = 1):
        generator_layer = [nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = generator_kernel_size,
                                    stride = generator_stride,
                                    bias = generator_bias)]
        generator_layer.append(nn.Tanh())
        return nn.Sequential(*generator_layer)
    
    def forward(self, latent_vector, label_vector):
        if not isinstance(latent_vector, torch.Tensor):
            latent_vector = torch.tensor(latent_vector, dtype=torch.float32).to(self.device)
        if not isinstance(label_vector, torch.Tensor):
            label_vector = torch.tensor(label_vector, dtype=torch.long).to(self.device)
        latent_vector = latent_vector.float().to(self.device)
        label_vector = label_vector.type(torch.LongTensor).to(self.device)
        batch_size = latent_vector.shape[0]
        print('label: ', label_vector.shape)
        print('latent: ',latent_vector.shape)
        latent_vector = self.latent_conditioning(latent_vector)
        print('label: ', label_vector.shape)
        label_vector = self.label_conditioning(label_vector)
        
        print('label: ', label_vector.shape)
        print('latent: ',latent_vector.shape)
        #reshape
        print(label_vector.shape)
        label_vector = label_vector[:,:,-1,:]
        print(label_vector.shape)
        label_vector = label_vector.view(batch_size,
                                        1,
                                        4,
                                        4)
        print('label: ', label_vector.shape)
        latent_vector = latent_vector.view(batch_size,
                                          1023,
                                          4,
                                          4)
        print('latent: ',latent_vector.shape)
        concatenated_input = torch.cat((label_vector, latent_vector), 1)
        print(concatenated_input.shape)
        concatenated_input = self.main_head(concatenated_input)
        print(concatenated_input.shape)
        concatenated_input = self.upsample_body(concatenated_input)
        print(concatenated_input.shape)
        concatenated_input = self.generated_img_Layer(concatenated_input)
        return concatenated_input