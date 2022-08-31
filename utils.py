# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:46:26 2022

@author: tanvi
"""


#%%get average for each epoch
class Averager:
    
    def __init__(self):
        
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        
        self.current_total = 0.0
        self.iterations = 0.0
        
#%%
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


