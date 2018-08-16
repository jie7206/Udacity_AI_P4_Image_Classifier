# Imports python modules
import argparse
import torch
import torch.nn.functional as F
import json
import time
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def get_input_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input image file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file name')
    parser.add_argument('-s','--save_dir', type=str, help='checkpoint save directory',default='save')
    parser.add_argument('-t','--top_k', type=int, help='return top k class name')
    parser.add_argument('-c','--category_names', type=str, help='the file which store category names of flowers',default='cat_to_name.json')
    parser.add_argument('-g','--gpu', help='GPU turned on/off', action="store_true")
    args = parser.parse_args()
    if args.gpu:
        print("GPU turned on")

    return args

def get_category_names(category_names):

    with open(category_names, 'r') as f:
        return json.load(f)

def load_checkpoint(save_dir, arch_name):

    checkpoint = torch.load("./{}/{}.pth".format(save_dir,arch_name))

    if arch_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch_name == 'densenet':
        model = models.densenet121(pretrained=True)
    elif arch_name == 'vgg':
        model = models.vgg16(pretrained=True)
 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path, resize_px = 256, crop_px = 224):
    
    # Scales image
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    
    if width > height:
        width = int(width * (resize_px/height))
        height = resize_px    
    else:
        height = int(height * (resize_px/width))
        width = resize_px
        
    img = img.resize((width,height))
    
    # Crops image        
    left = int((width-crop_px)/2)
    upper = int((height-crop_px)/2)
    right = left + crop_px
    lower = upper + crop_px
    
    img = img.crop((left,upper,right,lower))
    
    # Normalizes image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = np.array(img) / 255
    img = (img - mean)/std
    img = img.transpose((2, 0, 1))

    return img

def predict(image_path, model, gpu=False, topk=None):
    if gpu:
        model.cuda()
    else:
        model.cpu()

    data = torch.from_numpy(process_image("./{}".format(image_path)))
    data = data[np.newaxis, :].float()
    if gpu:
        data = data.cuda()
    output = model.forward(data)
    if gpu:
        output = output.cpu()
    ps = torch.exp(output).data
    if not topk:
        topk = 3
    tk = ps.topk(topk)
    probs = [round(x,8) for x in tk[0][0].numpy().tolist()]
    classes = tk[1][0].numpy().astype(np.str).tolist()
    
    return probs, classes