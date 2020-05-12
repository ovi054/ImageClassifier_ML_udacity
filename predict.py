'''
How to run:
command
python predict.py flowers/test/102/image_08042.jpg checkpoint2.pth --top_k 5
here image path and checkpoint path is mandatory
other is optional
'''

import argparse

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


import numpy as np
from collections import OrderedDict
from PIL import Image
import os
import pandas as pd
import seaborn as sns
import json
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help="Path of input image(Mandatory) | Example: flowers/test/102/image_08042.jpg")
    parser.add_argument('checkpoint', type = str, help="Path of checkpoint(Mandatory) | Example: checkpoint2.pth")
    parser.add_argument('--category_names', type = str, help="Optional) | Example: --category_names cat_to_name.json")
    parser.add_argument('--gpu', type = str, help="Example: --gpu Y (not Mandatory|By default Y" )
    parser.add_argument('--top_k', type = int, help="Example: --top_k 5 (not Mandatory|By default 1" )
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
     
    '''
    
    img = Image.open(image)
    transformation = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    return transformation(img)





def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if(topk == None):
        topk = 1
    model.to('cpu')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        
    probabilities = torch.exp(output)
    
    topk_probabilities, topk_labels = probabilities.topk(topk)
    
    return([topk_probabilities, topk_labels])



#Main Function
args = arg_parser()
print("~ Image-path: {}".format(args.image_path))    
img = args.image_path
checkpoint = args.checkpoint
model = load_checkpoint(checkpoint)
#print(model)
if args.category_names == None:
    cat_name = 'cat_to_name.json'
else:
    cat_name = args.category_names
with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)
    
if args.gpu == 'Y' or args.gpu == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print("Currently using: {}".format(device))
    model.to(device)
        


test_probabilities, test_labels = predict(args.image_path, model , args.top_k)

labels_list = test_labels.squeeze().tolist()
probabilities_list = test_probabilities.squeeze().tolist()
print('Top-k labels number:')
print(labels_list)
print('Top-k labels probability:')
print(probabilities_list)

text_labels_list = []
for label in labels_list:
    text_labels_list.append(cat_to_name[str(label)])
print('Top-k labels name:')
print(text_labels_list)


     

    
