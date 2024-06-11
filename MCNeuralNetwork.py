
import torch
import torch.nn as nn

class MCNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(12288,5000),
            nn.Sigmoid(),
            nn.Linear(5000,200),
            nn.Sigmoid(),
            nn.Linear(200,1),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, inputs):
        inputs = torch.FloatTensor(inputs)
        return self.model(inputs)
    
    def train(self, inputs, targets):
        targets = torch.FloatTensor(targets)
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets) # Error
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

