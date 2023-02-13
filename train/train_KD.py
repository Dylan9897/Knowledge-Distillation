"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/27 14:02
@Email : handong_xu@163.com
"""
import sys
sys.path.append('/mnt/d/资源/Github/知识蒸馏')

from module.TextCNN import *
from sklearn.metrics import classification_report

RANDOM_SEED = 2022
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_fn_kd(students_output, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(students_output / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(students_output, labels) * (1. - alpha)
    return KD_loss

def dev(model, data_loader):
    label2idx = {'教育': 0, '家居': 1, '时尚': 2, '时政': 3, '科技': 4, '房产': 5, '财经': 6}
    idx2label = {idx: label for label, idx in label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            cnn_ids = batch['x']
            labels = batch['label']
            logits = model(cnn_ids)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    table = classification_report(true_labels, pred_labels)
    return acc, table




