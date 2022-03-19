import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
class MetaNetwork(nn.Module):
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location='cpu'))
    
    def embed(self, input):
        embedding = self.encoder(input)
        return embedding

class biEncoder(MetaNetwork):

    def __init__(self, config):

        super(biEncoder, self).__init__()

        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=self.config['bias']),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 7, 2, bias=self.config['bias']),
            nn.Flatten(),
            nn.Linear(1024, 512, bias=self.config['bias']),
            nn.Linear(512, 64, bias=self.config['bias']),
        )

        self.coss = nn.CosineSimilarity(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.parameters(), lr=self.config['lr'])

    def forward(self, input):
        return self.encoder(input)

    def update(self, input, target):
        # Input shape -> [batch_size, 3 (triplet), 28, 28]
        anchors = input[:, 0, :, :].unsqueeze(1)
        positives = input[:, 1, :, :].unsqueeze(1)
        negatives = input[:, 2, :, :].unsqueeze(1)

        # Embedding
        anchors = self(anchors)
        positives = self(positives)
        negatives = self(negatives)

        # Cross similarity pairs
        a2p = self.coss(anchors, positives).unsqueeze(1)
        a2n = self.coss(anchors, negatives).unsqueeze(1)

        # Cross entropy loss
        self.optimizer.zero_grad()
        prediction = torch.cat([a2p, a2n], dim=1)
        loss = self.loss(prediction, target)
        loss.backward()
        self.optimizer.step()

        # Accuracy 
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1).detach().cpu().numpy()
        accuracy = f1_score(np.zeros_like(prediction), prediction, zero_division=1)
        return loss, accuracy

    def evaluate(self, input, target):
        # Input shape -> [batch_size, 3 (triplet), 28, 28]
        anchors = input[:, 0, :, :].unsqueeze(1)
        positives = input[:, 1, :, :].unsqueeze(1)
        negatives = input[:, 2, :, :].unsqueeze(1)

        # Embedding
        anchors = self(anchors)
        positives = self(positives)
        negatives = self(negatives)

        # Cross similarity pairs
        a2p = self.coss(anchors, positives).unsqueeze(1)
        a2n = self.coss(anchors, negatives).unsqueeze(1)

        # Cross entropy loss
        prediction = torch.cat([a2p, a2n], dim=1)
        loss = self.loss(prediction, target)
        # Accuracy 
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1).detach().cpu().numpy()
        accuracy = f1_score(np.zeros_like(prediction), prediction)
        return loss, accuracy

if __name__ == '__main__':

    arr = torch.rand((30, 3, 28, 28)).cuda()

    config = {
        'bias' : True,
        'lr' : 1e-3
    }
    model = biEncoder(config)
    model.cuda()

    model.update(arr)
