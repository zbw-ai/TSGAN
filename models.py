# 定义模型结构
# 0. 导入库
import torch
from torch import nn


# 1. Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.batch = nn.BatchNorm1d(30, affine=False)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)   # reshape 3D
        self.linear = nn.Linear(hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()   # [0,1]  

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
        # X -> (0,1)
        X = self.batch(X)
        o, h = self.GRU(X)
        o = self.linear(o)
        # o = self.sigmoid(o)  # [0, 1]
        return o


# 2. Discriminator
class Discriminator(nn.Module):
    '''
        to use BCEWithLogitsLoss instead of BCELoss. 
        Why? For numerical stability: see the first paragraph of 
      https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html.
    '''
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.batch = nn.BatchNorm1d(31, affine=False)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1) 
        self.sigmoid = nn.Sigmoid()   # [0,1]
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
        # X -> (0,1)
        X = self.batch(X)
        o, h = self.GRU(X)
        o = self.linear(o)
        o = self.sigmoid(o)
        return o


# import torch
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU

# sequence_length = 30
# z_size = 24       # 输入噪声维度
# latent_size = 24  # 隐藏层维度
# num_layers = 1    # 网络深度
# feature_size = 5

# # Generate noise vector  暂时不需要噪声
# Z = torch.rand(feature_size, sequence_length, z_size).to(device)
# print(Z.shape)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad