import torch
import torch.nn as nn





class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)

        # self.bn1    = nn.BatchNorm1d(1024)
        # self.bn2    = nn.BatchNorm1d(512)
        # self.bn3    = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.5)

        self.out    = nn.Linear(128, 39)
        self.act_fn = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.act_fn(x)
        
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.act_fn(x)
        
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.act_fn(x)
        
        x = self.out(x)
        
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, is_bidirection=True, num_layers=4, dropout_rate=0.1):

        super(LSTMNet, self).__init__()
        self.hidden_dim    = hidden_dim
        self.is_bidirection = is_bidirection
        self.num_directions = 2 if is_bidirection else 1
        self.num_layers     = num_layers

        self.lstm   = nn.LSTM(input_dim, self.hidden_dim, num_layers=num_layers, dropout=dropout_rate, bidirectional=is_bidirection, batch_first=True)
        self.l1 = nn.Linear(self.hidden_dim*self.num_directions, 256) 
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 39)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.shape[0]

        hidden_init = torch.ones(self.num_layers*self.num_directions, batch_size, self.hidden_dim).cuda()
        cell_init   = torch.ones(self.num_layers*self.num_directions, batch_size, self.hidden_dim).cuda()

        output, (hidden_state, cell_state) = self.lstm(x.cuda(), (hidden_init, cell_init))
        output = self.l1(output)
        output = self.l2(output)
        output = self.l3(output)
        return output

# BLSTM for pBLSTM
# Step 1: Reduce time resolution to half
# Step 2: Run through BLSTM
# Note the input should have timestamp%2 == 0
# class pBLSTMLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
#         super(pBLSTMLayer, self).__init__()
#         self.rnn_unit = getattr(nn, rnn_unit.upper())

#         # feature dimension will be doubled since time resolution reduction
#         self.BLSTM = self.rnn_unit(input_dim*2, hidden_dim, 1, bidirectional=True, dropout=dropout_rate, batch_first=True)

#     def forward(self, input):
#         batch_size = input.size(0)
#         timestep = input.size(1)
#         feature_dim = input.size(2)

#         # Reduce time resolution
#         input = input.contiguous().view(batch_size, int(timestep/2), feature_dim*2)

#         # Bidirectional RNN
#         hidden_state, cell_state = self.BLSTM(input)
#         return hidden_state, cell_state


# # Listener is a pBLSTM(pyramial Bi-LSTM) stacking 3 layer to reduce time resolution 8 times
# class Listener(nn.Module):
#     def __init__(self, input_dim, hidden_dim, listener_layer, rnn_unit, dropout_rate, **kwargs):
#         super(Listener, self).__init__()

#         # Listener RNN layer
#         self.listener_layer = listener_layer
#         assert self.listener_layer >= 1, 'Listener should have at least 1 layer'

#         self.pLSTM_layer0 = pBLSTMLayer(input_dim, hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate)

#         for i in range(1, self.listener_layer):
#             setattr(self, 'pLSTM_layer'+str(i), pBLSTMLayer(hidden_dim*2, hidden_dim, rnn_unit=rnn_unit, dropout_rate=dropout_rate))




# if __name__ == '__main__':

    

