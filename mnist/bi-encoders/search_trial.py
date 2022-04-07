import torch
import torch.nn as nn
from models import biEncoder
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image
import argparse

class Seeker:

    def __init__(self, model, database, device):
        self.database = database
        self.model = model
        self.device = device
        self.similarity_fn = nn.CosineSimilarity(dim=1)

    def __call__(self, query, top_k=25):
        query = query.to(self.device).float()
        query_embedding = self.model.embed(query.unsqueeze(0))

        # Similarity 
        self.similarity = self.similarity_fn(query_embedding, self.database)
        self.similarity = [(str(sim), str(i)) for i, sim in enumerate(self.similarity.cpu().detach().numpy())]
        self.similarity = sorted(self.similarity, key=lambda x: x[0], reverse=True)

        return self.similarity[:top_k]


    

def plot(query, results, save_path, query_id=0):

    # Query image
    query_img = Image.fromarray(query.squeeze().numpy(), mode='L')
    # Results image
    
    results_img = [Image.fromarray(result.squeeze().numpy(), mode='L') for result in results]

    # Plots
    fig, axs = plt.subplots(6, 5)
    
    # Remove frames from plots
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    # plt.subplot(6, 5, 1)
    axs[0, 2].imshow(query_img)
    axs[0, 2].set_title("QUERY")
    
    
    for i, result_img in enumerate(results_img):
        if i//5 +1 == 2 and i%5 ==0:
            axs[i//5 +1 , i%5].set(ylabel='RESULTS')
        if i//5 +1 == 3 and i%5 ==0:
            axs[i//5 +1 , i%5].set(ylabel='TOP {}'.format(len(results_img)))
        axs[i//5 +1 , i%5].imshow(result_img)
    plt.savefig('results/{}/{}'.format(save_path, query_id))
    plt.show()

def main(id, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert device == 'cuda', "CUDA FAILED?"

    mnist_data_train = MNIST('../data/', train=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    mnist_data_test = MNIST('../data/', train=False, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))]))
    # trainloader = DataLoader(mnist_data_train, batch_size=1, shuffle=4, num_workers=4)
    # testloader = DataLoader(mnist_data_test, batch_size=1, shuffle=False, num_workers=4)
    config = {
        'bias' : True,
        'lr' : 0
    }
    model = biEncoder(config)
    model.load('saved_models/biencoder2')
    model.to(device)
    img_db = torch.load('embeddings/embeddings2.pt')

    search_engine = Seeker(model, img_db, device)

    query_id = id
    query = mnist_data_train[query_id][0]
    # query = torch.flatten(query)
    q = mnist_data_train.data[query_id]
    results = search_engine(query, top_k=25)
    
    r = [mnist_data_test.data[i[1]] for i in results]

    plot(q, r, save_path, query_id=query_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=int)
    parser.add_argument('--save_path',type=str)

    args = parser.parse_args()

    main(args.id, args.save_path)
