import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

import argparse

import json

parser = argparse.ArgumentParser(description = " This file is used to predict the image classifier ")

parser.add_argument('path_to_img', action = "store",type=str)
parser.add_argument('checkpoint', action = "store", default = "./checkpoint.pt",type=str)
parser.add_argument('--top_k', action = "store", default = 5,dest = 'top_k',type=int)
parser.add_argument('--category_names', action = "store", default = "cat_to_name.json",dest='cat_to_name_path',type=str)
parser.add_argument('--gpu', action = "store_true",dest = 'use_gpu')

def load_checkpoint(path,device):
    checkpoint = torch.load(path)
    model = eval(f'models.{checkpoint["model"]}(pretrained=True)')
    model.classifier =  nn.Sequential(checkpoint['layers'])
    model.classifier.load_state_dict(checkpoint['state'])
    model.class_to_idx = checkpoint['class_to_idx']
    # This is a more useful form because I get idx from topk and I want the class
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    model.to(device)
    return model
def get_transforms():
     return transforms.Compose([transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])])
def process_image(image_path,transforms):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image_path)
    im = transforms(im)
    return im
def predict(image_path, model,device,transforms, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        im = process_image(image_path,transforms)
        im.resize_(1,*im.size())
        im = im.to(device)
        model = model.to(device)
        ps = torch.exp(model.forward(im))
        ps= ps.cpu()
        top_ps, top_classes = ps.topk(topk, dim=1)
        top_classes =[model.idx_to_class[_] for _ in top_classes.cpu().numpy().squeeze().tolist()]
        return top_ps.cpu().numpy().squeeze(), top_classes    
def main(args):
    if args.use_gpu and not torch.cuda.is_available() : 
        print("GPU requested but not available using CPU instead")
        device = 'cpu'
    elif not args.use_gpu:
        device = 'cpu'
    else:
        device = 'cuda'
    model = load_checkpoint(args.checkpoint,device)
    probs, top_classes = predict(args.path_to_img, model,device, get_transforms(), args.top_k)
    with open(args.cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[_] for _ in top_classes]
    for i,(p,name) in enumerate(zip(probs,labels)):
        print(f"Top class no. {i+1} is {name} with probability {p*100}%")

if __name__ == "__main__":
    main(parser.parse_args())
