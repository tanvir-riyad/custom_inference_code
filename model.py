# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:37:06 2022

@author: tanvir
"""


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def detection_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

