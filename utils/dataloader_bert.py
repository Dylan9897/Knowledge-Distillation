"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/29 13:05
@Email : handong_xu@163.com
"""
"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/16 11:03
@Email : handong_xu@163.com
"""

from utils.utils import *
import torch
from torch.utils.data import Dataset, DataLoader



class ContentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        #         print(encoding['input_ids'])
        return {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df,tokenizer,max_len,batch_size):
    ds = ContentDataset(
        texts = df['content'].values,
        labels = df['label'].values,
        tokenizer = tokenizer,
        max_len = max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        # num_worders = 4
    )



if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams

    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

    rcParams['figure.figsize'] = 12, 8

    from transformers import BertModel, BertTokenizer, AdamW
    filePath =DATA_DIR + "/labeled_data.csv"
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE)
    content,label = read_file(filePath,analyse=True)
    print(label)
    tokens_len = []
    for txt in content:
        tokens = tokenizer.encode(txt,max_length=512)
        tokens_len.append(len(tokens))
    sns.displot(tokens_len)
    plt.xlim([0, 256])
    plt.ylim([0,30])
    plt.xlabel('Token count')
    plt.savefig(ROOT_DIR+'/png/length.png')





