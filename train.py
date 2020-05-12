'''
How to run:
command
python train.py flowers --arch vgg16 --learning_rate 0.001 --epochs 10
Here data directory: flowers is mandatory. Other is optional.
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

def arg_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--limit', default=5, type=int)
    parser.add_argument('data_dir', type = str, help="Root Directory of images" )
    parser.add_argument('--arch', type = str, help="Example: --arch vgg16 / --arch vgg13 (not Mandatory|By default vgg16" )
    parser.add_argument('--learning_rate', type = float, help="Example: --learning_rate 0.001 (not Mandatory|By default 0.001" )
    parser.add_argument('--hidden_units', type = int, help="Example: --hidden_units 1024 (not Mandatory|By default 1024" )
    parser.add_argument('--epochs', type = int, help="Example: --epochs 10 (not Mandatory|By default 10" )
    parser.add_argument('--gpu', type = str, help="Example: --gpu Y (not Mandatory|By default Y" )
    args = parser.parse_args()
    return args


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets

    data_transforms = { 'train': transforms.Compose([transforms.RandomRotation(30),
                                                     transforms.RandomResizedCrop(224),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])]),
                       'valid' : transforms.Compose([transforms.Resize(255),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])]),
                       'test' : transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

                      }


    # TODO: Load the datasets with ImageFolder
    image_datasets = { 'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])


                     }


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
                   'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)


                  }
    print("Data loading complete.")
    return dataloaders['train'], dataloaders['valid'], dataloaders['test'], image_datasets['train']

def tarin_model(trainloader, validloader, testloader,img_dataset_train , arch='vgg16',learning_rate=0.001,hidden_units=1024, epochs=10,gpu='Y'):
    
    if arch==None:
        arch = 'vgg16'
    if learning_rate==None:
        learning_rate=0.001
    if hidden_units==None:
        hidden_units =1024
    if epochs==None:
        epochs = 10
    if gpu == None:
        gpu = 'Y'
    print("obvi",arch,learning_rate, epochs)  
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p = 0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
    else:
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p = 0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if gpu=='Y':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Currently using: {}".format(device))
        model.to(device)
    #print(model)
    steps = 0
    running_loss = 0
    print_every = 50
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {test_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("\nTraining Completed")
    model.class_to_idx = img_dataset_train.class_to_idx
    checkpoint = { 'state_dict': model.state_dict(),
                   'classifier': model.classifier,
                   'class_to_idx': model.class_to_idx,
                   'epochs': epochs,
                   'arch': arch}

     
    torch.save(checkpoint, 'checkpoint2.pth')   
    print(checkpoint['arch'])
        
#Main Function
args = arg_parser()
print("~ data-dir: {}".format(args.data_dir))
trainloader, validloader, testloader , img_dataset_train = load_data(args.data_dir)
tarin_model(trainloader, validloader, testloader, img_dataset_train, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu )
