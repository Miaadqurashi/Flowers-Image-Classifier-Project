import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser(description = " This file is used to train the image classifier ")

parser.add_argument('data_dir', action = "store", default = "./flowers",type=str)
parser.add_argument('--save_dir', action = "store", default = "./checkpoint.pth",dest='save_dir')
parser.add_argument('--arch', action = "store", default = "vgg11",dest = 'arch')
parser.add_argument('--learning_rate', action = "store", default = 0.001,dest = 'lr',type=float)
parser.add_argument('--hidden_units1', action = "store", default = 2048,dest = 'hu1',type=int)
parser.add_argument('--hidden_units2', action = "store", default = 512,dest = 'hu2',type=int)
parser.add_argument('--epochs', action = "store", default = 512,dest = 'epochs',type=int)
parser.add_argument('--gpu', action = "store_true",dest = 'use_gpu')
def get_data_loaders_and_transforms(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                       'valid/test': transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])}


    image_datasets = {'train': datasets.ImageFolder(train_dir,transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir,transform=data_transforms['valid/test']),
                      'test': datasets.ImageFolder(test_dir,transform=data_transforms['valid/test'])}


    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                  'valid': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
                  'test': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)}
    return data_transforms, image_datasets, dataloaders
def get_model(arch,hu1,hu2):
    if arch.startswith('densenet'):
        mapping = {'densenet121':1024,'densenet161':2208,'densenet169':1664,'densenet201':1920}
        try:
            input_shape = mapping[arch]
        except KeyError : 
            print('Wrong Densenet passed the supported architectures by PyTorch are: densenet121, densenet161, densenet169, and densenet201')
            return None
    else: 
        if arch not in ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']:
            print("Wrong VGg archteture passed available options:'vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg16','vgg16_bn','vgg19','vgg19_bn'")
            return None
        input_shape = 25088
    model = eval(f'models.{arch}(pretrained=True)')
    for param in model.parameters():
        param.requires_grad = False


    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_shape, hu1)),
                              ('relu1', nn.ReLU()),
                              ('Dp1',nn.Dropout(p=0.15)),
                              ('fc2', nn.Linear(hu1, hu2)),
                              ('relu2', nn.ReLU()),
                              ('Dp2',nn.Dropout(p=0.15)),
                              ('fc3', nn.Linear(hu2, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        
    model.classifier = classifier
    return model
def train_model(model,optimizer,criterion,dataloaders,epochs=15,device='cpu'):
    steps = 0
    running_loss = 0
    print_every = 50
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
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
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Valid accuracy: {accuracy*100/len(dataloaders['valid']):.3f}%")
                running_loss = 0
                model.train()
              
def save_checkpt(save_dir,model,arch, class_to_idx):
    checkpoint = {"model": arch, ## Will use eval to load it.
               'class_to_idx' :class_to_idx,
               'layers' : OrderedDict([each for each in model.classifier.named_children()]),
               'state' : model.classifier.state_dict()               
              }
    torch.save(checkpoint, save_dir)

 
def main(args):
    data_transforms, image_datasets, dataloaders = get_data_loaders_and_transforms(args.data_dir)
    if not ( args.arch.startswith('vgg') or args.arch.startswith('densenet') ):
        print("Only VGG and densenet (all variants)  models are supported")
        return
    model = get_model(args.arch,args.hu1,args.hu2)
    if model is None:
        return
    if args.use_gpu and not torch.cuda.is_available() : 
        print("GPU requested but not available using CPU instead")
        device = 'cpu'
    elif not args.use_gpu:
        device = 'cpu'
    else:
        device = 'cuda'
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model.to(device)
    train_model(model,optimizer,criterion,dataloaders,args.epochs,device)
    save_checkpt(args.save_dir, model, args.arch, image_datasets['train'].class_to_idx)
            
    

if __name__ == "__main__":
    main(parser.parse_args())
