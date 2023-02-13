"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/27 11:11
@Email : handong_xu@163.com
"""

"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/3/4 14:09
@Email : handong_xu@163.com
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/mnt/d/资源/Github/TextCNN')
from module.Mish import Mish
from utils.utils import *

class Config(object):
    def __init__(self,path,embedding):
        self.EMBED_ROOT = EMBED_DIR
        self.activate = None
        self.model_name = 'TextCNN'

        # self.file_path = os.path.join(path,'labeled_data.csv')

        self.class_list = ['教育','家居','时尚','时政','科技','房产','财经']

        self.vocab_path = DATA_DIR+"vocab.pkl"
        self.save_path = MODEL_DIR
        self.log_path = os.path.join(path,'log')
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(self.EMBED_ROOT,'embedding_SougouNews.npz'))["embeddings"].astype('float32')
        ) if embedding!='random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000 # 若超过1000个batch效果没有提升，则提前结束训练
        # self.num_classes = len(self.class_list)
        self.num_classes = 7
        self.n_vocab = 0
        self.num_epoches = 2
        self.batch_size = 4
        # self.pad_size = 32
        self.learning_rate = 1e-2
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2,3,4)
        self.num_filters = 256
        self.init_method = False
        self.optimizer = False
        self.moment = 0.8
        self.alpha = 0.9

class Model(nn.Module):
    def __init__(self,config=None):
        super(Model, self).__init__()
        self.mish =Mish()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab,config.embed,padding_idx=config.n_vocab-1)
        self.dropout = nn.Dropout(config.dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,config.num_filters,(k,config.embed)) for k in config.filter_sizes]
        )
        self.fc = nn.Linear(config.num_filters*len(config.filter_sizes),config.num_classes)

    def conv_and_pool(self,x,conv):
        x = conv(x)
        x =self.mish(x).squeeze(3)
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        out = self.embedding(x) # torch.Size([128,32,422])
        out = out.unsqueeze(1) # torch.Size([128,1,32,422])
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1) # torch.Size
        out = self.dropout(out)  # torch.Size([128,768])
        out = self.fc(out) #torch.Size([128,10])
        return out








