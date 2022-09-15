# Data preprocessing

# We set sequence_length = 10~30 minutes.

# 1. 导入库
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 2. 定义参数

# 定义时间序列长度 
sequence_length = 30
# 输入地址
data_path = "/data112/zengbw/Code_MSRA/Dataset/Stock_debug/sz300007.csv"

# 提取窗口
def extract_windows(data, sequence_length):
    '''
        Extract rolling windows of length sequence_length
        from the data

        Input: numpy array
        Returns: torch tensor
    '''
    X = []
    Y = []
    scaler = MinMaxScaler()
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        x_new = pd.DataFrame()
        x_new['close'] = x['close']   # 直接用收盘价close
        # x_new['close'] = (x['close']-x['close'].iloc[0]) /x['close'].iloc[0]  # 转化为变化率

        # for _ in ['close', 'open', 'high', 'low']:
        #     x_new[_] = (x[_]-x[_].iloc[0]) /x[_].iloc[0]
        # x_new['amount'] = x['amount']/x['amount'].mean()  # 成交量比较特殊
        y = data[i:i + sequence_length + 1]   # get Y
        y_new = pd.DataFrame()
        y_new['close'] = y['close']
        # y_new['close'] = (y['close']-y['close'].iloc[0]) /y['close'].iloc[0]

        # scaled_x = scaler.fit_transform(x_new)  
        scaled_x = np.array(x_new)

        # scaled_y = scaler.fit_transform(y_new)
        scaled_y = np.array(y_new)

        X.append(scaled_x.astype(np.float32))
        Y.append(scaled_y.astype(np.float32))
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def data_preprocess(data_path, sequence_length):
    # 读取csv
    paths_raw = pd.read_csv(data_path)    # [58080, 7]

    # 取数据（特征）
    paths_new = pd.DataFrame()
    paths_new["close"] = paths_raw["close"]
    # paths_new["open"] = paths_raw["open"]
    # paths_new["high"] = paths_raw["high"]
    # paths_new["low"] = paths_raw["low"]
    # paths_new["amount"] = paths_raw["amount"]

    # 取固定天数的数据(one day)
    paths = paths_new[0:2420]    # [242, 5]  class: dataframe

    # 数据预处理:提取滑动窗口
    X, Y = extract_windows(paths, sequence_length)   # torch.Size([212, 30, 5])   torch.Tensor

    return X, Y

X, Y = data_preprocess(data_path, sequence_length)
print(X.shape, Y.shape)










