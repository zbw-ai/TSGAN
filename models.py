# 定义模型结构
# 0. 导入库
import torch
from torch import nn


# 1. Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)   # hidden_size -> 1
        # self.sigmoid = nn.Sigmoid()   # [0,1]     实际上应该是到[-0.1, 0.1]

        """
        interval_min = -840
        interval_max = 840
        scaled_mat = (sample_mat - np.min(sample_mat) / (np.max(sample_mat) - np.min(sample_mat)) * (interval_max - interval_min) + interval_min
        """

        '''
        Initialise weights; this is copied verbatim from
        https://github.com/d9n13lt4n/timegan-pytorch and is supposed 
        to mimic the weight initialisation used in the tensorflow implementation.
        Full credit to the author.
        '''
        with torch.no_grad():
            for name, param in self.GRU.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X):
        '''
        Args:
            X: minibatch of input timeseries 
            (batch_size, sequence_length, input_size)
        Returns:
            o: minibatch of timeseries 
            (batch_size, sequence_length, hidden_size)
        '''
        o, h = self.GRU(X)
        o = self.linear(o)
        # o = self.sigmoid(o)
        return o


# 2. Discriminator
class Discriminator(nn.Module):
    '''
        For some reason, 
        implementations do not use sigmoid activation for the discriminator,
        and they use BCEWithLogitsLoss instead of BCELoss. 
        Why? For numerical stability: see the first paragraph of 
      https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html.
    '''
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1) 

        '''
        Initialise weights; this is copied verbatim from
        https://github.com/d9n13lt4n/timegan-pytorch and is supposed 
      to mimic the weight initialisation used in the tensorflow implementation.
        Full credit to the author.
        '''
        with torch.no_grad():
            for name, param in self.GRU.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, X):
        '''
        Args:
      X: minibatch of input timeseries (batch_size, sequence_length, input_size)
        Returns:
      o: minibatch of timeseries (batch_size, sequence_length, hidden_size)
        '''
        o, h = self.GRU(X)
        o = self.linear(o)
        return o
