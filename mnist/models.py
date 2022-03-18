from audioop import bias
from numpy import average
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score


class MetaNetwork(nn.Module):
    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path, map_location='cpu'))
    
    def embed(self, input):
        embedding = self.encoder(input)
        return embedding
class MNIST_classifier(MetaNetwork):

    def __init__(self, config):
        super(MNIST_classifier, self).__init__()
        
        self.config = config
        self.encoder = nn.Sequential(
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
        embedding = self.encoder(input)
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





class AutoEncoder(MetaNetwork):
    def __init__(self,config):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(784, 256,bias=self.config['bias']),
            nn.ReLU(),
            nn.Linear(256, 128,bias=self.config['bias']),
            nn.ReLU(),
            nn.Linear(128, 64,bias=self.config['bias']),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128,bias=self.config['bias']),
            nn.ReLU(),
            nn.Linear(128, 256,bias=self.config['bias']),
            nn.ReLU(),
            nn.Linear(256, 784,bias=self.config['bias']),
            #nn.ReLU(),
        )

        self.loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters(),lr=self.config['lr'])
    

    def forward(self,input,is_decode=False):
        x = torch.flatten(input, start_dim=1)
        
        encoded = self.encoder(x)
        
        if is_decode:
            decoded = self.decoder(encoded)
            return encoded,decoded
        return encoded
    
    def update(self,input,test=False):
        # encoded, decoded = self(input,is_decode=True)
        
        self.optimizer.zero_grad()
        x = torch.flatten(input, start_dim=1)
        encoder_outputs = []
        for layer in self.encoder.children():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                # print(x.shape)
                encoder_outputs.append(x)

        decoder_outputs = []
        for layer in self.decoder.children():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                # print(x.shape)
                decoder_outputs.append(x)

        encoder_outputs.pop()
        #decoder_outputs.pop()
        decoder_outputs.reverse()
        
        recons_loss = self.loss(x,torch.flatten(input,start_dim=1))
        #inter_loss = torch.autograd.Variable(torch.tensor(0.0),requires_grad=False)
        # inter_loss = torch.tensor(0.0).cuda()
        # for i, j in zip(encoder_outputs, decoder_outputs):
        #     inter_loss+=self.loss(i,j)
        
        # total_loss = recons_loss+inter_loss
        total_loss = recons_loss

        
        if test==False:
            total_loss.backward()
            self.optimizer.step()

        return total_loss

if __name__ == '__main__':
    config = {
        'bias' : True,
        'lr' : 1e-3,
    }
    
    model = AutoEncoder(config)

    arr = torch.rand((5, 1, 28, 28)).float()
    for i in range(10):
        print(model.update(arr).item())

