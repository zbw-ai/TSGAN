# 2022.09.12
# bulid a timegan model from scratch

# 0. 指定GPU
import torch
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

# plt.rcParams["figure.figsize"] = (10, 5) # 指定plot的图片规格

# 2. 定义参数

# 定义时间序列长度 
sequence_length = 30
# 输入地址
data_path = "/data112/zengbw/Code_MSRA/Dataset/Stock_debug/sz300007.csv"


# dataset
# input:X   shape:[数据条数, 序列长度, 特征维度]
X = data_preprocess(data_path, sequence_length)
X = X.to(device)

