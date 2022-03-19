import numpy as np 
import torch
import os, sys
from tqdm import tqdm
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import biEncoder
from torch.utils.data import Dataset

class tripletSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[idx]



def biencoder_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert device == 'cuda', "CUDA FAILED?"

    mnist_data_train = torch.load('data/triplet_ds_train.pt')
    mnist_data_train = tripletSet(mnist_data_train)
    mnist_data_test = torch.load('data/triplet_ds_test.pt')
    mnist_data_test = tripletSet(mnist_data_test)

    trainloader = DataLoader(mnist_data_train, batch_size=32, shuffle=4, num_workers=4)
    testloader = DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=4)
    config = {
        'bias' : True,
        'lr' : 3e-4,
    }

    model = biEncoder(config)
    model.to(device)

    # TRAINING

    losses, accuracies = [], []
    
    for epoch in tqdm(range(15)):
        total_loss = 0
        total_accuracy = 0
        c = 0
        for batch in tqdm(trainloader):
            batch = batch.to(device).float()
            targets = torch.zeros(batch.shape[0])
            targets = targets.to(device).long()
            loss, accuracy = model.update(batch, targets) 
            losses.append(loss)
            total_loss += loss.item()
            total_accuracy += accuracy
            c += 1
    
        print("train loss: ", total_loss/c)
        print("train accuracy: ", total_accuracy/c)

    
    
    test_losses,test_accuracies = [],[]
    total_loss, total_accuracy = 0, 0
    
    # TEST
    t = 0
    for batch in tqdm(testloader):
        batch = batch.to(device).float()
        targets = torch.zeros(batch.shape[0])
        targets = targets.to(device).long()
        with torch.no_grad():
            loss, accuracy = model.evaluate(batch,targets)
            losses.append(loss)
            total_loss += loss.item()
            total_accuracy += accuracy
            t+=1
    

    print("Test loss:", total_loss/t)
    print("train accuracy: ", total_accuracy/t)

    
    model.save("saved_models/biencoder2")



if __name__ == '__main__':
    biencoder_training()