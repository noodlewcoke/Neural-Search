import torch
from models import biEncoder
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader
from training import tripletSet

def build_biencoder():
    config = {
        'bias' : True,
        'lr' : 1e-3,
    }
    model = biEncoder(config)
    model.load("saved_models/biencoder2")    
    model.cuda()
    mnist_data_test = MNIST('../data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))

    testloader = DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=4)
    
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for imgs,labels in testloader:
            imgs = imgs.cuda()
            embeddings = model.embed(imgs)
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    embeddings_list = torch.cat(embeddings_list,dim=0)
    labels_list = torch.cat(labels_list,dim=0)

    torch.save(embeddings_list,"embeddings/embeddings2.pt")
    torch.save(labels_list,"embeddings/labels2.pt")


if __name__ == "__main__":
    build_biencoder()