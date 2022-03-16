from audioop import bias
from numpy import average
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score


class MNIST_classifier(nn.Module):

    def __init__(self, config):
        super(MNIST_classifier, self).__init__()
        
        self.config = config
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=self.config['bias']),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 7, 2, bias=self.config['bias']),
            nn.Flatten(),
            nn.Linear(1024, 512, bias=self.config['bias']),
            nn.Linear(512, 64, bias=self.config['bias']),
        )
        self.classifier = nn.Linear(64, 10)

        self.optimizer = Adam(self.parameters(), lr=self.config['lr'])
        self.loss = nn.CrossEntropyLoss()
        # Layers 


    def forward(self, input):
        embedding = self.network(input)
        logits = self.classifier(embedding)
        return logits, embedding

    def update(self, input, target):
        logits, _ = self(input)
        self.optimizer.zero_grad()
        loss = self.loss(logits, target)
        loss.backward()
        self.optimizer.step()

        outputs = F.softmax(logits, dim=-1)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = f1_score(target.cpu().long().numpy(), outputs.cpu().float().detach().numpy(), average='macro')

        return loss, accuracy

    def evaluate(self, input, target):
        logits, _ = self(input)
        loss = self.loss(logits, target)

        outputs = F.softmax(logits, dim=-1)
        outputs = torch.argmax(outputs, dim=-1)
        accuracy = f1_score(target.cpu().long().numpy(), outputs.cpu().float().detach().numpy(), average='macro')

        return loss, accuracy

    def embed(self, input):
        embedding = self.network(input)
        return embedding

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location='cpu'))



if __name__ == '__main__':
    config = {
        'bias' : True,
        'lr' : 1e-3,
    }
    
    model = MNIST_classifier(config)

    arr = torch.rand((5, 1, 28, 28)).float()
    t = torch.tensor([1,2,3,4,5]).long()
    output = model.update(arr, t)
    print(output)

