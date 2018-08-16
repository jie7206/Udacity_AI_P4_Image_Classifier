#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Lin Shih Chieh
# DATE CREATED: 2018-07-25
# 目的: 使用 train.py 用数据集训练新的网络
#   基本用途： python train.py data_directory
#   在训练网络时，输出训练损失、验证损失和验证准确率
#   选项：
#     设置保存检查点的目录： python train.py data_dir ‐‐save_dir save_directory
#     选择架构： python train.py data_dir ‐‐arch "vgg13"
#     设置超参数： python train.py data_dir ‐‐learning_rate 0.01 ‐‐hidden_units 512 ‐‐epochs 20
#     使用 GPU 进行训练： python train.py data_dir ‐‐gpu
#   Example call:
#    python train.py flowers ‐‐save_dir save ‐‐arch vgg ‐‐learning_rate 0.01 ‐‐hidden_units 512 ‐‐epochs 20 ‐‐gpu

# Imports python modules
import argparse
import torch
import torch.nn.functional as F
import os
import json
import time
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def main():

    # Define get_input_args() function to create & retrieve command line arguments
    args = get_input_args()

    # Set gobal params
    data_dir = "./{}/".format(args.data_directory)
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'

    # Prepare Dataset then load for train and valid
    train_data = set_data(data_dir+train_dir,rotation_flip=True)
    valid_data = set_data(data_dir+valid_dir)
    test_data = set_data(data_dir+test_dir)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    # Set model architecture : pick any of the following : vgg, alexnet, densenet
    model = init_model(args.arch, args.hidden_units)

    ########## BEGIN training model ##########

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    running_loss = 0
    print_every = 5

    if args.gpu:
        model.cuda()
    else:
        model.cpu()

    for ei in range(args.epochs):

        for ti, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)

            if args.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if ti > 0 and ti % print_every == 0:
                    model.eval()
                    valid_accuracy = 0
                    valid_loss = 0
                    for vi, (inputs, labels) in enumerate(validloader):
                        inputs, labels = Variable(inputs), Variable(labels)
                        if args.gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()
                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        valid_accuracy += equality.type_as(torch.FloatTensor()).mean() 
                    print("Epochs: {:.0f} ".format(ei+1),
                          "Train Loss: {:.4f} ".format(running_loss/print_every),
                          "Valid Loss: {:.4f} ".format(valid_loss/len(validloader)),
                          "Valid Accuracy: {:.4f}".format(valid_accuracy/len(validloader)))
                    running_loss = 0
                    model.train()

    ########## END training model ##########

    # Save training checkpoint args.save_dir, args.arch
    path = "./{}".format(args.save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    f = open("{}/{}.pth".format(args.save_dir,args.arch), 'wb')
    save_checkpoint(model, train_data, f)

def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help='set data directory for training')
    parser.add_argument('-s','--save_dir', type=str, help='checkpoint save directory',default='save')
    parser.add_argument('-a','--arch', type=str, help='model: pick any of the following : vgg(default), alexnet, densenet',default='vgg')
    parser.add_argument('-l','--learning_rate', type=float, help='set learning rate value for training',default=0.001)
    parser.add_argument('-u','--hidden_units', type=int, help='set hidden units number for training',default=1024)
    parser.add_argument('-e','--epochs', type=int, help='set epochs number for training',default=3)
    parser.add_argument('-g','--gpu', help='GPU turned on/off', action="store_true")
    args = parser.parse_args()
    if args.gpu:
        print("GPU turned on")

    return args

def set_data(data_path, rotation_flip=False):

    transforms_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if rotation_flip:
        transforms_list = [transforms.RandomRotation(30), transforms.RandomHorizontalFlip()] + transforms_list
    data_transforms = transforms.Compose(transforms_list)

    return datasets.ImageFolder(data_path, transform=data_transforms)

def init_model(arch_name, hidden_units):

    if arch_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier_in_features = 9216
    elif arch_name == 'densenet':
        model = models.densenet121(pretrained=True)
        classifier_in_features = 1024
    else:
        model = models.vgg16(pretrained=True)
        classifier_in_features = 25088

    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(classifier_in_features, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))      
    model.classifier = classifier
    print("{} 网络模型加载完成！".format(arch_name))

    return model

def save_checkpoint(model, train_data, f):
    checkpoint_dict = { 'classifier': model.classifier,
                        'class_to_idx': train_data.class_to_idx,
                        'state_dict': model.state_dict() }
    torch.save(checkpoint_dict, f)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("网络模型文档于 {} 存档完成!".format(now))

# Call to main function to run the program
if __name__ == "__main__":
    main()