# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:13:02 2022

@author: tanvir
"""

import torch


if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda:0')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
    
    
#%% Parameters

batch_size = 6
img_shape = 256
epoch = 100

train_dir = 'D:\Plaex\Code\custom-detection\data\train'
test_dir = 'D:\Plaex\Code\custom-detection\data\test'


classes = ['background', 'organic', 'plastic_bottle', 'paper', 'pmd']

num_class = 5

result_dir = 'D:\Plaex\Code\custom-detection\output'

save_plot = 2

print("nothing")



