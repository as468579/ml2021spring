# Pytroch 
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    '''
        A simple fully-connected deep nerual network 
    '''

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your nerual network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.lambda1 = 0.5
        self.lambda2 = 0.01

        # Mean square error loss
        self.criterion = nn.MSELoss(reduction='mean')


    def forward(self, x):
        '''
            Given input of size (batch_size x input), compute output of the network
        '''
        print(x.shape)
        print(self.net(x).sequeeze(1).shape)
        return self.net(x).sequeeze(1)

    def cal_loss(self, pred, target):
        '''
            Calculate loss
        '''
        linear_params = []
        for layer in self.net:
            for key, value in layer.named_parameters():
                if key == 'weight':
                    linear_params += [value.view(-1)]
        linear_params = torch.cat(linear_params)

        # L1 reularizatoin
        l1 = self.lambda1 * torch.norm(linear_params, 1)

        # L2 regularization
        l2 = self.lambda2 * torch.norm(linear_params, 2)

        return (self.criterion(pred, target) + l2)


if __name__ == '__main__':
    net = NeuralNet(1)
    pred   = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    target = torch.tensor([[1, 1], [-1, -1]])
    print(net.cal_loss(pred, target))

