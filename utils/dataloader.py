"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/28 13:30
@Email : handong_xu@163.com
"""
"""
本代码实现的功能实现一个集成TextCNN和Bert的Dataloader
"""
import os
import sys
sys.path.append('/mnt/d/资源/Github/知识蒸馏')

import time
import jieba
import torch
import pickle as pkl
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.utils import DATA_DIR
from module.bert import tokenizer

MAX_VOCAB_SIZE = 10000  # 词表长度限制
RANDOM_SEED = 2022
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
VOCAB_PATH = "vocab_{}.pkl"
PAD_SIZE= 256 # padding size
FILE_PATH = os.path.join(DATA_DIR,'labeled_data.csv')
BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def cws(seq):
    return ' '.join(jieba.cut(seq))

def read_file(file,analyse=False):
    df = pd.read_csv(file)
    if analyse:
        print(f'value counts is {df["class_label"].value_counts()}')
        print(df['class_label'].describe())
    label_id2cate = dict(enumerate(df.class_label.unique()))
    label_cate2id = {value: key for key, value in label_id2cate.items()}
    df['label'] = df['class_label'].map(label_cate2id)
    return df


def build_vocab(data,tokenize,max_size,min_freq):
    """
    :param data: list of content
    :param tokenize: operate content function
    :param max_size: max length of vocabs
    :param min_freq: min vocab freq
    :return: word2idx,type:dict
    """
    vocab_dic = {}
    for content in tqdm(data):
        for word in tokenize(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

def build_dataset(use_word=None):
    df = read_file(FILE_PATH)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    if use_word:
        print('Use word level ...')
        tokenize = lambda x: cws(x).split(' ')  # 以空格隔开，word-level
        vocab_path = os.path.join(DATA_DIR,VOCAB_PATH.format('word'))
    else:
        print('use char level ...')
        tokenize = lambda x: [y for y in x]  # char-level
        vocab_path = os.path.join(DATA_DIR,VOCAB_PATH.format('char'))
    print('load vocab')
    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path,'rb'))
    else:
        vocab = build_vocab(df['content'].values,tokenize,MAX_VOCAB_SIZE,min_freq=1)
        pkl.dump(vocab,open(vocab_path,'wb'))
    print('loading vocab done...')
    def load_dataset(data,pad_size=PAD_SIZE):
        contents = []
        for i in data.index:

            content = data.loc[i]['content']
            label = data.loc[i]['label']
            token = tokenize(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            contents.append((token,int(label),seq_len))
        return contents

    train = load_dataset(df_train)
    dev = load_dataset(df_val)
    test = load_dataset(df_test)
    return vocab,train,dev,test

class KD_Dataset(object):
    def __init__(self,vocab):
        self.word2idx = vocab
        self.index = 0
        self.device = device

    def word_to_idx(self,seq):
        word_line = []
        for word in seq:
            word_line.append(self.word2idx.get(word,self.word2idx.get(UNK)))
        return word_line

    def _to_tensor(self,text):
        x = torch.LongTensor(text).to(self.device)
        return x

    def encode_fn(self,text):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

    def encode(self,batches):
        result = {
            'texts':[],
            'input_ids':[],
            'attention_mask':[],
            'x':[],
            'label':[],
            'lengths':[]
        }
        for unit in batches:
            result['texts'].append(''.join(unit[0]))
            result['x'].append(self.word_to_idx(unit[0]))
            result['label'].append(unit[1])
            result['lengths'].append(unit[2])
        result['x'] = self._to_tensor(result['x'])
        result['label'] = self._to_tensor(result['label'])
        result['lengths'] = self._to_tensor(result['lengths'])
        return result

    def test(self,seqList):
        encoding = self.encode_fn(seqList)
        print(encoding['input_ids'].shape)

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,vocab):
        self.fun = KD_Dataset(vocab)
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device



    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self.fun.encode(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self.fun.encode(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset,vocab):
    start_time = time.time()
    print(f'start to create dataiters , now is {start_time}')
    iter = DatasetIterater(dataset, BATCH_SIZE, device,vocab)
    print(f'finish create dataiters , cost time is {get_time_dif(start_time)}s')
    return iter


if __name__ == '__main__':
    param = ['今天天气真好','哈喽','我的天啊','你在干嘛']
    fun = KD_Dataset({UNK:0})
    fun.test(param)






