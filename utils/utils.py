"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/16 10:56
@Email : handong_xu@163.com
"""
import os
import sys
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR,'data/长文本分类')
MODEL_DIR = os.path.join(ROOT_DIR,'ckpt')
LOG_DIR = os.path.join(ROOT_DIR,'logs')
BERT_BASE = os.path.join(MODEL_DIR,'bert-base-chinese')
EMBED_DIR = DATA_DIR

def write_pkl_file(obj,path):
    with open(path,'wb') as ft:
        pickle.dump(obj,ft)

def read_pkl_file(path):
    with open(path,'rb') as fl:
        obj = pickle.load(fl)
    return obj



if __name__ == '__main__':
    print(ROOT_DIR)
    print(DATA_DIR)
    print(MODEL_DIR)
    print(LOG_DIR)