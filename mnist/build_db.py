import torch
from models import MNIST_classifier, AutoEncoder
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader

def build_classifier():
    config = {
        'bias' : True,
        'lr' : 1e-3,
    }
    model = MNIST_classifier(config)
    model.load("saves/vanilla")    
    model.cuda()
    mnist_data_test = MNIST('data/', train=False, transform=torchvision.transforms.Compose([
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

    torch.save(embeddings_list,"embeddings/embeddings.pt")
    torch.save(labels_list,"embeddings/labels.pt")


def build_ae():
    config = {
        'bias' : True,
        'lr' : 1e-3,
    }
    model = AutoEncoder(config)
    model.load("saves/vanilla_ae")    
    model.cuda()
    mnist_data_test = MNIST('data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))

    testloader = DataLoader(mnist_data_test, batch_size=1000, shuffle=False, num_workers=4)
    
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for imgs,labels in testloader:
            imgs = torch.flatten(imgs,start_dim=1)
            imgs = imgs.cuda()
            embeddings = model.embed(imgs)
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    embeddings_list = torch.cat(embeddings_list,dim=0)
    labels_list = torch.cat(labels_list,dim=0)

    torch.save(embeddings_list,"embeddings/ae_embeddings.pt")
    torch.save(labels_list,"embeddings/ae_labels.pt")

if __name__ == "__main__":
    build_ae()