import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from module.Ranger import opt_func
from module.utils import get_time_dif
from tensorboardX import SummaryWriter
from transformers import BertModel, BertTokenizer, AdamW



def train(config,model,train_iter,dev_iter,test_iter):
    start_time = time.time()
    model.train()
    if config.model_name == "TextCNN":
        optim = opt_func(model.parameters(),lr=config.learning_rate)
    else:
        optim = AdamW(model.parameters())
    total_batch = 0
    dev_best_loss = float("inf")
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime("%m-%d_%H.%M",time.localtime()))
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch+1,config.num_epochs))
        for i,unit in enumerate(train_iter):
            if config.model_name == "TextCNN":
                feature = unit["embed"].to(config.device)
                labels = unit['labels'].to(config.device)
                outputs = model(feature)
            else:
                input_ids = unit["input_ids"].to(config.device)
                attention_mask = unit["attention_mask"].to(config.device)
                labels = unit["labels"].to(config.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            loss = F.cross_entropy(outputs,labels)
            print(outputs)
            print(labels)
            print(loss)
            s = input()
            loss.backward()
            optim.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path.format('best'))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path.format('best')))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, unit in enumerate(data_iter):
            if config.model_name == "TextCNN":
                feature = unit["embed"].to(config.device)
                labels = unit['labels'].to(config.device)
                outputs = model(feature)
            else:
                input_ids = unit["input_ids"].to(config.device)
                attention_mask = unit["attention_mask"].to(config.device)
                labels = unit["labels"].to(config.device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        target_names = ['教育','家居','时尚','时政','科技','房产','财经']
        report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)        





