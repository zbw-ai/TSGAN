# 2022.09.12
# bulid a timegan model from scratch

# 0. 指定GPU
import torch
# import os
# print(os.environ['CUDA_VISIBLE_DEVICES'])
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU


# 1. 导入库
# %matplotlib inline
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
torch.manual_seed(0) 

from sklearn.preprocessing import MinMaxScaler
from d2l import torch as d2l
import random
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
# import statsmodels.api as sm


# 导入自己写的库
from dataset import data_preprocess
from models import Generator, Discriminator

# plt.rcParams["figure.figsize"] = (10, 5) # 指定plot的图片规格

# 2. 定义参数

# 定义时间序列长度 
sequence_length = 30
# 输入地址
data_path = "/data112/zengbw/Code_MSRA/Dataset/Stock_debug/sz300007.csv"

batch_size = 128  # 批量大小

z_size = 24       # 输入噪声维度

latent_size = 24  # 隐藏层维度

num_layers = 1    # 网络深度

lr = 1e-4         # 学习率

# loss
L1_loss = nn.L1Loss().to(device)
MSE_loss = nn.MSELoss().to(device)
BCE_loss = nn.BCEWithLogitsLoss().to(device)

# dataset
# input:X   shape:[数据条数, 序列长度, 特征维度]
X, Y = data_preprocess(data_path, sequence_length)
X = X.to(device)
Y = Y.to(device)

input_size = X.shape[-1]  # 属性个数(即特征维度)

# DataLoader
train_data = TensorDataset(X, Y)   # Y: torch.Size([212, 31, 1])
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
# drop_last:对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
# 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点

# 查看维度: Shape of input (X) and output (y)
for _X_, y in train_loader:
    print(_X_.shape, y.shape)   # torch.Size([128, 30, 5]) torch.Size([128,b31, 1])
    break


# 实例化网络
G = Generator(z_size, latent_size, num_layers)
D = Discriminator(latent_size, latent_size, num_layers) 

# 送入GPU
G = G.to(device)
D = D.to(device)

# 优化器
G_optimiser = torch.optim.Adam(G.parameters(), lr=lr)
D_optimiser = torch.optim.Adam(D.parameters(), lr=lr)

# 重置梯度
def reset_gradients():
    G.zero_grad()
    D.zero_grad()

# def build_GAN(G, D):
#     x = G()

# 定义训练过程
def train_D(X, Y):
    # # Generate noise vector  暂时不需要噪声
    # Z = torch.rand(X.shape[0], sequence_length, z_size).to(device)

    # 更新D
    # 1. 训练前梯度置零
    reset_gradients()
    # 2. 前向传播
    y_pred = G(X.float()).detach()

    # 将预测的 y_pred 替换掉原有序列的最后一个
    Y_fake = Y

    real_logit = D(Y)
    fake_logit = D(Y)
    # 3. 计算loss
    D_loss = BCE_loss(real_logit, torch.ones_like(real_logit).to(device)) + BCE_loss(fake_logit, torch.zeros_like(fake_logit).to(device))
    # 4. loss反向传播
    D_loss.backward()
    # 5. 优化器
    D_optimiser.step()

    return D_loss



def train_G(X, Y):
    # 更新G
    # 1. 训练前梯度置零
    reset_gradients()
    # 2. 前向传播
    y_pred = G(X.float())
    Y_fake = Y
    real_logit = D(Y)
    fake_logit = D(Y)
    # 3. 计算loss
    G_ad_loss = MSE_loss(fake_logit, torch.ones_like(fake_logit).to(device))
    G_p_loss = L1_loss(Y, Y_fake)     # p = 1
    # G_dp_loss = L1_loss(sgn(y_t+1 - y_t), sgn(y_pred - y_t))
    G_loss = G_ad_loss + G_p_loss
    # 4. loss反向传播
    G_loss.backward()
    # 5. 优化器
    G_optimiser.step()

    return G_loss



def train_joint(train_loader, num_epochs):
    G_losses = []
    D_losses = []
    for epoch in range(1, num_epochs+1):
        for X, Y in train_loader:
            
            for _ in range(2):
                '''
                Train discriminator half as often as 
                the generator and autoencoder
                '''
                G_loss = train_G(X, Y)
            
            D_loss = train_D(X, Y)

        G_losses.append(G_loss)
        D_losses.append(D_loss)
    
    return G_losses, D_losses


G_losses, D_losses = train_joint(train_loader, 20)







        



