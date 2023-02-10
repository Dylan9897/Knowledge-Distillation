import torch
import numpy as np
import pandas as pd
import argparse
import time
from importlib import import_module
from module.dataloader import create_data_loader,read_file,build_vocab,build_dataset
from module.utils import init_network
from module.train import train

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True  #


parser = argparse.ArgumentParser(description="知识蒸馏")
parser.add_argument("--task",default="TextCNN",type=str,help="choose a task:TextCNN,Bert,KD")
args = parser.parse_args()



if __name__=="__main__":
    # embedding = "embedding_SougouNews.npz"
    embedding = "random"
    df = read_file("data/labeled_data.csv")
    x = import_module('models.'+args.task)
    config = x.Config('',embedding)
    
    vocab,df_train,df_val,df_test = build_dataset(config)
    trainloader = create_data_loader(df_train,vocab,config)
    validloader = create_data_loader(df_val,vocab,config)
    testloader = create_data_loader(df_test,vocab,config)
    print("data loading done")

    
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # config.device = torch.device('cpu')
    start_time = time.time()
    config.n_vocab = len(vocab)
    print(config.n_vocab)
    config.init_method = "xavier"
    model = x.Model(config).to(config.device)
    # print(model.state_dict()["embedding.weight"].shape)
    # s = input()
    if args.task == "TextCNN":
        print("start init model")
        init_network(model,config)
    train(config,model,trainloader,validloader,testloader)


    



    

