import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
import torch.nn.functional as F
from module.Mish import Mish


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Bert'
        self.class_list = ["财经","房产","家居","教育","科技","时尚","时政"]                              # 类别名单
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.BERT_BASE = "ckpt/bert-base-chinese"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # self.device = torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.MAX_VOCAB_SIZE = 10000
        self.padding_size = 256                                         # 每句话处理成的长度(短填长切)
        self.batch_size = 4
        self.RANDOM_SEED = 2022
        self.file_path =dataset+"./data/labeled_data.csv"

        self.train_path = dataset + './data/train.csv'                                # 训练集
        self.dev_path = dataset + './data/dev.csv'                                    # 验证集
        self.test_path = dataset + './data/test.csv'                                  # 测试集
        self.class_list = ["财经","房产","家居","教育","科技","时尚","时政"]                              # 类别名单
        self.vocab_path = dataset + './data/vocab.pkl'                                # 词表
        self.save_path = dataset + './saved_dict/' + self.model_name +"{}"+ '.ckpt'        # 模型训练结果
        self.log_path = dataset + './log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('ckpt/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量

        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.dropout = 0.5
        self.filter_sizes = (2,3,4)
        self.num_filters = 256
        self.embed = 768
        self.num_classes = 7

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.BERT_BASE)
        self.mish = Mish()
        self.dropout = nn.Dropout(self.config.dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,self.config.num_filters,(k,self.config.embed)) for k in self.config.filter_sizes]
        )
        self.fc = nn.Linear(self.config.num_filters*len(self.config.filter_sizes),self.config.num_classes)

    def conv_and_pool(self,x,conv):
        x = x.unsqueeze(1)
        x = conv(x)
        x = self.mish(x).squeeze(3)
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids,attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        out = torch.cat([self.conv_and_pool(_,conv) for conv in self.convs],1 )
        out = self.dropout(out)
        out = self.fc(out)
        return out





