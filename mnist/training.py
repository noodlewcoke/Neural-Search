import numpy as np 
import torch
import os, sys
from tqdm import tqdm
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import MNIST_classifier, AutoEncoder

def Classifier_Training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert device == 'cuda', "CUDA FAILED?"

    mnist_data_train = MNIST('data/', train=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    mnist_data_test = MNIST('data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    trainloader = DataLoader(mnist_data_train, batch_size=32, shuffle=4, num_workers=4)
    testloader = DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=4)
    # print(mnist_data_train.data.shape)

    config = {
        'bias' : True,
        'lr' : 1e-3,
    }

    model = MNIST_classifier(config)
    model.to(device)

    # TRAINING

    losses, accuracies = [], []
    
    for epoch in tqdm(range(8)):
        total_loss, total_accuracy = 0, 0
        c = 0
        for batch in tqdm(trainloader):
            images, labels = batch
            images = images.to(device).float()
            labels = labels.to(device).long()
            loss, accuracy = model.update(images, labels) 
            losses.append(loss)
            accuracies.append(accuracy)
            total_loss += loss.item()
            total_accuracy += accuracy
            c += 1
    
        print("train loss: ",total_loss/c)
        print("train acc: ",total_accuracy/c)
    
    
    test_losses,test_accuracies = [],[]
    total_loss, total_accuracy = 0, 0
    t = 0
    
    # TEST
    for y in tqdm(testloader):
        imgs,labels=y
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()
        with torch.no_grad():
            loss,accuracy=model.evaluate(imgs,labels)
            losses.append(loss)
            accuracies.append(accuracy)
            total_loss += loss.item()
            total_accuracy += accuracy
            t+=1
    

    print("Test acc:", total_accuracy/t)
    
    model.save("saves/vanilla")



def Ae_Training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert device == 'cuda', "CUDA FAILED?"

    mnist_data_train = MNIST('data/', train=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    mnist_data_test = MNIST('data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    trainloader = DataLoader(mnist_data_train, batch_size=32, shuffle=4, num_workers=4)
    testloader = DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=4)
    # print(mnist_data_train.data.shape)

    config = {
        'bias' : True,
        'lr' : 1e-4,
    }

    model = AutoEncoder(config)
    model.to(device)

    # TRAINING

    losses, accuracies = [], []
    
    for epoch in tqdm(range(20)):
        total_loss = 0
        c = 0
        for batch in tqdm(trainloader):
            images, labels = batch
            images = images.to(device).float()
            loss = model.update(images) 
            losses.append(loss)
            total_loss += loss.item()
            c += 1
    
        print("train loss: ",total_loss/c)
    
    
    test_losses = []
    total_loss = 0
    t = 0
    
    # TEST
    for y in tqdm(testloader):
        imgs,labels=y
        imgs = imgs.to(device).float()
        with torch.no_grad():
            loss=model.update(imgs,test=True)
            losses.append(loss)
            total_loss += loss.item()
            t+=1
    

    print("Test loss:", total_loss/t)
    
    model.save("saves/vanilla_ae")

if __name__ == '__main__':
    Ae_Training()