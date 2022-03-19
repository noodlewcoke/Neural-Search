import torch 
import numpy as np
import os 
from tqdm import tqdm
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pprint import pprint

def triplet_former(dataset, train=True):
    
    # Fetch classes separately
    data = dataset.data
    labels = dataset.targets
    
    triplet_set = []
    label_triplets = []
    print("Generating triplet dataset. (a, p, n)")
    for i in tqdm(np.arange(10)):
        # Extract class indices
        indices = np.where(labels == i)[0]
        other_indices = np.where(labels != i)[0]

        # Compose triplets
        for anchor in indices:
            # Select a positive match
            positive_match = np.random.choice(indices, replace=False)          # Do not select the same indice twice
            # Select a negative match
            negative_match = np.random.choice(other_indices, replace=False)    # The same idea

            assert labels[positive_match] == labels[anchor] and labels[anchor] != labels[negative_match], f"{labels[anchor]}, {labels[positive_match]}, {labels[negative_match]}"
            label_t = [labels[anchor], labels[positive_match], labels[negative_match]]
            label_triplets.append(label_t)


            data_sample = torch.cat([
                data[int(anchor)].unsqueeze(0), 
                data[int(positive_match)].unsqueeze(0), 
                data[int(negative_match)].unsqueeze(0)], 
                dim=0)

            triplet_set.append(data_sample)

    triplet_set = torch.stack(triplet_set, dim=0)
    print("Triplet dataset shape: ", triplet_set.shape)

    if train:
        torch.save(triplet_set, 'data/triplet_ds_train.pt')
    else:
        torch.save(triplet_set, 'data/triplet_ds_test.pt')

    # pprint(label_triplets[:100])

if __name__ == "__main__":
    mnist_data_train = MNIST('../data/', train=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    mnist_data_test = MNIST('../data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))

    triplet_former(mnist_data_train, train=True)
    triplet_former(mnist_data_test, train=False)
