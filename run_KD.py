"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/29 13:47
@Email : handong_xu@163.com
"""

import sys
sys.path.append('/mnt/d/资源/Github/知识蒸馏')

from importlib import import_module
from utils.dataloader import build_iterator,read_file,build_dataset
from utils.dataloader_bert import create_data_loader
from sklearn.model_selection import train_test_split
from module.bert import tokenizer,TextCNN_Classifier

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from train.train_KD import *
from train.train_TextCNN import init_network

MAX_LEN = 256
RANDOM_SEED = 2022
BATCH_SIZE = 4
EPOCHES = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = import_module('module.' + 'TextCNN')
config = x.Config(DATA_DIR,'random')

print('loading textCNN data')
vocab, textcnn_train_data, textcnn_dev_data, textcnn_test_data = build_dataset()
textcnn_train_iter = build_iterator(textcnn_train_data,vocab)
textcnn_dev_iter = build_iterator(textcnn_dev_data,vocab)
textcnn_test_iter = build_iterator(textcnn_test_data,vocab)
print('finish loading textCNN data')

print('loading bert data')
train = read_file(DATA_DIR+'/labeled_data.csv')
df_train,df_test = train_test_split(train,test_size=0.1, random_state=RANDOM_SEED)
df_val,df_test = train_test_split(df_test,test_size=0.5,random_state=RANDOM_SEED)
train_data_loader = create_data_loader(df_train,tokenizer,MAX_LEN,BATCH_SIZE)
val_data_loader = create_data_loader(df_val,tokenizer,MAX_LEN,BATCH_SIZE)
test_data_loader = create_data_loader(df_test,tokenizer,MAX_LEN,BATCH_SIZE)
print('finish loading bert data')

print('start to init textcnn model')
config.n_vocab = len(vocab)
textcnn = x.Model(config).to(config.device)
init_network(textcnn,config)
textcnn.train()
print('text cnn model init done')

print('star to loading bert best model')
MODEL_PATH = os.path.join(MODEL_DIR,'best_model_state.ckpt')
bert = TextCNN_Classifier()
bert.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
bert.to(device)
bert.eval()
print('bert best model loading done')

# 冻结bert参数
for name, p in bert.named_parameters():
    p.requires_grad = False

optimizer = Adam(textcnn.parameters(), lr=2e-5)

print('start training ...')
best_acc = 0.
for epoch in range(EPOCHES):
    for i,(cnndata,bertdata) in enumerate(zip(textcnn_train_iter,train_data_loader)):
        optimizer.zero_grad()
        labels = cnndata['label']
        students_output = textcnn(cnndata['x'])

        input_ids = bertdata["input_ids"].to(device)
        attention_mask = bertdata["attention_mask"].to(device)
        teacher_output = bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn_kd(students_output,labels,teacher_output,10,0.9)
        loss.backward()
        optimizer.step()

        # 打印信息
        if i % 100 == 0:
            labels = labels.data.cpu().numpy()
            preds = torch.argmax(students_output, dim=1)
            preds = preds.data.cpu().numpy()
            acc = np.sum(preds == labels) * 1. / len(preds)
            print("TRAIN: epoch: {} step: {} acc: {} loss: {} ".format(epoch + 1, i, acc, loss.item()))

        acc, table = dev(textcnn, textcnn_dev_iter)
        print("DEV: acc: {} ".format(acc))
        # print("DEV classification report: \n{}".format(table))

        if acc > best_acc:
            torch.save(textcnn.state_dict(), MODEL_DIR+'/best_kd.ckpt')
            best_acc = acc

    # print("start testing ......")
    # test_loader = DataLoader(KDdataset(config.base_config.test_data_path), batch_size=config.batch_size, shuffle=False)
    # best_model = textcnn()
    # best_model.load_state_dict(torch.load(config.model_path))
    acc, table = dev(textcnn, textcnn_test_iter)

    print("TEST acc: {}".format(acc))
    print("TEST classification report:\n{}".format(table))