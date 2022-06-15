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
        self.current_epoch+=1
        decay_std_dev = self.total_epochs * self.decay_ratio
        current_std_dev = 1.0 - self.current_epoch/decay_std_dev
        current_std_dev *= self.initial_std_dev - self.min_std_dev
        current_std_dev+=self.min_std_dev
        print(current_std_dev, self.min_std_dev, )
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
                 hidden_channels = (512, 256, 128, 64, 32),
                 final_linear = 1,
                 stride = 2,
                 kernel_size = 2,
                 bias = False
                ):
        super(discriminator_conv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = 1
        self.total_epochs = total_epochs
        self.tree = nn.Sequential(*[
            self.discriminator_body(hidden_channels[i], hidden_channels[i + 1]) for i in range(len(hidden_channels) - 1)])
        self.root_score = self.root(331776)
        self.to(self.device)
        

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
    
    def forward(self, input_):
        if not isinstance(input_, torch.Tensor):
            input_ = torch.tensor(input_, dtype=torch.float32).to(self.device)
        input_ = self.tree(input_)
        batch_size = input_.shape[0]
        accumilated_flattening_size = input_.shape[1]*input_.shape[2]*input_.shape[3]
        input_ = torch.flatten(input_)
        input_ = input_.view(batch_size, accumilated_flattening_size)
        #print(input_.shape)
        input_ = self.root_score(input_)
        return input_

    
    
    
    
    
    
    

class generator_upsample(nn.Module):
    def __init__(self,
                latent_dim,
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
        self.latent_size = latent_dim
        self.main_head = self.head()
        self.upsample_body = (nn.Sequential(*
                                  [self.body(hidden_channels[i],
                                   hidden_channels[i + 1]) for i in range(len(hidden_channels) - 1)]
                                 ))
        self.generated_img_Layer = self.generate_layer(hidden_channels[-1], 
                                                                      out_channel_generator)
        self.to(self.device)
        
        
    def head(self):
        head_layers = [nn.Conv2d(in_channels = 1,
                                 out_channels = self.hidden_channels_head,
                                 kernel_size = self.kernel_size_head,
                                 stride = self.stride_head,
                                 bias = False
                                )
                      ]
        head_layers.append(nn.LazyBatchNorm2d())
        head_layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*head_layers)
    
    def body(self,
            in_channels,
            out_channels,
            dropout = True,
            dropout_rate = 0.05,
            kernel_body_size = 4,
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
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        #print(x.shape)
        x_head = self.main_head(x)
        x_head = self.upsample_body(x_head)
        x_head = self.generated_img_Layer(x_head)
        return x_head