import torch
import torch.nn as nn

import time
from datetime import timedelta
import pickle


# 获取已使用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
def init_network(model, config, exclude='embedding', seed=123):
    print('init model......')
    print(config.init_method)
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if config.init_method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif config.init_method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def write_pkl_file(obj,path):
    with open(path,'wb') as ft:
        pickle.dump(obj,ft)

def read_pkl_file(path):
    with open(path,'rb') as fl:
        obj = pickle.load(fl)
    return obj



