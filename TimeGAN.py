# 2022.09.12
# bulid a timegan model from scratch

# 2022.09.18
# change according meeting at 09.15


# 0. 指定GPU
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU

# 1. 导入库
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns

# import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
torch.manual_seed(0) 

# from sklearn.preprocessing import MinMaxScaler
# from d2l import torch as d2l
import random
# from sklearn.manifold import TSNE 
# from sklearn.decomposition import PCA
# import statsmodels.api as sm

# 导入自己写的库
from dataset import data_preprocess
from models import Generator, Discriminator, set_requires_grad

plt.rcParams["figure.figsize"] = (10, 5) # 指定plot的图片规格

# 2. 定义参数

# 定义时间序列长度 
sequence_length = 30
# 输入地址
data_path = "/data112/zengbw/Code_MSRA/Dataset/Stock_debug/sz300007.csv"

batch_size = 128  # 批量大小

# z_size = 24       # 输入噪声维度   (用不上)

latent_size = 24  # 隐藏层维度

num_layers = 3   # 网络深度

lr = 1e-4         # 学习率

# loss
L1_loss = nn.L1Loss().to(device)
MSE_loss = nn.MSELoss().to(device)
BCE_loss = nn.BCEWithLogitsLoss().to(device)

# dataset
# input:X   shape:[数据条数, 序列长度, 特征维度]
X, Y = data_preprocess(data_path, sequence_length)
# X = X.squeeze(-1)
# Y = Y.squeeze(-1)
X = X.to(device)
Y = Y.to(device)

input_size = X.shape[-1]  # 属性个数(即特征维度)  （暂时不用）

# DataLoader
train_data = TensorDataset(X, Y)   # Y: torch.Size([212, 31])
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
# drop_last:对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
# 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点

# 查看维度: Shape of input (X) and output (y)
for _X_, y in train_loader:
    print("X.shape, y.shape:", _X_.shape, y.shape)   # torch.Size([128, 30, 1]) torch.Size([128, 31, 1])
    break


# 实例化网络
Gen = Generator(input_size, latent_size, num_layers)
Dis = Discriminator(input_size, latent_size, num_layers) 

# 送入GPU
Gen = Gen.to(device)
Dis = Dis.to(device)

# 优化器
G_optimiser = torch.optim.Adam(Gen.parameters(), lr=lr)
D_optimiser = torch.optim.Adam(Dis.parameters(), lr=lr)

# 重置梯度
def reset_gradients():
    Gen.zero_grad()
    Dis.zero_grad()


# 定义训练过程
def train_D(X, Y):
    # # Generate noise vector  暂时不需要噪声
    # Z = torch.rand(X.shape[0], sequence_length, z_size).to(device)
    set_requires_grad([Dis], True)  
    set_requires_grad([Gen], False) 
    # 更新D
    # 1. 训练前梯度置零
    reset_gradients()
    # 2. 前向传播
    y_pred = Gen(X.float()).detach()  # 防止更新G

    # 将预测的 y_pred 替换掉原有序列的最后一个
    Y_fake = torch.cat((Y[:,:-1], y_pred), 1) 
    real_logit = Dis(Y)  
    fake_logit = Dis(Y_fake)

    # 3. 计算loss
    D_loss = BCE_loss(real_logit, torch.ones_like(real_logit).to(device)) + BCE_loss(fake_logit, torch.zeros_like(fake_logit).to(device))
    #
    D_optimiser.zero_grad()
    # 4. loss反向传播
    D_loss.backward()
    # 5. 优化器
    D_optimiser.step()

    return D_loss



def train_G(X, Y):
    set_requires_grad([Gen], True)  
    set_requires_grad([Dis], False) 
    # 更新G
    # 1. 训练前梯度置零
    reset_gradients()
    # 2. 前向传播
    y_pred = Gen(X.float())
    Y_fake = torch.cat((Y[:,:-1], y_pred), 1) 
    # real_logit = D(Y)  
    fake_logit = Dis(Y_fake)
    # 3. 计算loss
    G_ad_loss = MSE_loss(fake_logit, torch.ones_like(fake_logit).to(device))
    G_p_loss = L1_loss(Y[:,-1], Y_fake[:,-1])     # p = 1   # 还真的会除以序列长度31
    # G_dp_loss = L1_loss(torch.sgn(Y[:,-1] - Y[:,-2]), torch.sgn(Y_fake[:,-1] - Y[:,-2]))  # torch.sign(a)
    G_loss = G_p_loss + 100 * G_p_loss  #  + G_dp_loss
    #
    G_optimiser.zero_grad()
    # 4. loss反向传播
    G_loss.backward()
    # 5. 优化器
    G_optimiser.step()

    return G_loss, G_ad_loss, G_p_loss



def train_joint(train_loader, num_epochs):
    G_losses = []
    G_ad_losses = []
    G_p_losses = []
    D_losses = []
    print(" Start to train G and D !")
    for epoch in tqdm(range(1, num_epochs+1)):
        for X, Y in train_loader:
            
            for _ in range(2):
                '''
                Train discriminator half as often as 
                the generator and autoencoder
                '''
                G_loss, G_ad_loss, G_p_loss = train_G(X, Y)
            
            D_loss = train_D(X, Y)

        G_losses.append(G_loss)
        G_ad_losses.append(G_ad_loss)
        G_p_losses.append(G_p_loss)
        D_losses.append(D_loss)
    
    return G_losses, D_losses


G_losses, D_losses = train_joint(train_loader, 100)

# loss显示
G_losses1 = [i.detach().cpu() for i in  G_losses]
D_losses1 = [i.detach().cpu() for i in  D_losses]

plt.plot(G_losses1)
plt.savefig("G_losses.png")

plt.plot(D_losses1)
plt.savefig("D_losses.png")


print("Train finished!")


############################
# save and load model parammeters

# Save model parameters

save = "__base__"
torch.save(Gen.state_dict(), f"/data112/zengbw/Code_MSRA/Results/trained_models/G{save}")
torch.save(Dis.state_dict(), f"/data112/zengbw/Code_MSRA/Results/trained_models/D{save}")

# Load model parameters
load = "__base__"

Gen.load_state_dict(torch.load(f"/data112/zengbw/Code_MSRA/Results/trained_models/G{save}"))
Dis.load_state_dict(torch.load(f"/data112/zengbw/Code_MSRA/Results/trained_models/D{save}"))


# Evaluation
def generate(X):
    Y_pred = X.float()
    for i in range(20):
        y_pred = Gen(Y_pred[:,-30:].float()).detach()   # 取倒数30个数
        # 将预测的 y_pred 替换掉原有序列的最后一个
        Y_pred = torch.cat((Y_pred, y_pred), 1)   
        # Y_pred = torch.cat((Y[:,:-1], y_pred), 1) 
    return Y_pred


"""
# 对于多变量数据
def stack(X):
    '''
    (Helper function)
    Since TimeGAN was trained on a multivariate series,
    we stack the data so it is as if we had univariate 
    series 
    '''
    X_stacked = []
    for multivariate_window in X:
        for variate in range(input_size):
            X_stacked.append(multivariate_window.T[variate])
    X_stacked = np.array(np.vstack(X_stacked))

    return X_stacked
"""


generated_data = generate(X).to(device).detach().cpu().numpy()

# Visual Inspection
# Plot a randomly selected generated window
plt.title("A randomly selected generated window")
plt.plot(np.array(generated_data)[random.sample(list(np.arange(generated_data.shape[0])), 1)[0]])
plt.savefig("fake_data.png")

# Plot a randomly selected real window
plt.title("A randomly selected real window")
plt.plot(np.array(X.cpu())[random.sample(list(np.arange(X.shape[0])), 1)[0]])
plt.savefig("real_data.png")


print("Test finished !")

        




# %%
