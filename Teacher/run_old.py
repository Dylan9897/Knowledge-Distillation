import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from models.bert import TextCNN_Classifier,Config,tokenizer
from sklearn.metrics import confusion_matrix, classification_report
from module.dataloader import build_dataset,create_data_loader
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging



parser = argparse.ArgumentParser(description="Long Text Classification")
parser.add_argument('--optim',default=False,help="choose differential learning rate or not")
parser.add_argument('--epoch',default=5,help='num of epoches')
parser.add_argument('--batch',default=4,help='num of batchsize')
args = parser.parse_args()

MAX_LEN = 256
BATCH_SIZE = args.batch
RANDOM_SEED = 2022
EPOCHS = args.epoch
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
config.RANDOM_SEED = RANDOM_SEED
config.file_path = "data/labeled_data.csv"
config.vocab_path = "Student/data/vocab.pkl"
config.padding_size =256
config.MODEL_DIR = "Teacher/saved_dict"


# (df,tokenizer,max_len,batch_size)
vocab,df_train,df_val,df_test = build_dataset(config)

train_data_loader = create_data_loader(df_train,tokenizer,MAX_LEN,BATCH_SIZE)
val_data_loader = create_data_loader(df_val,tokenizer,MAX_LEN,BATCH_SIZE)
test_data_loader = create_data_loader(df_test,tokenizer,MAX_LEN,BATCH_SIZE)
config.n_examples = len(tra)
model = TextCNN_Classifier()
model = model.to(device)


if not args.optim:
    optimizer = AdamW(model.parameters(),lr=2e-5,correct_bias=False)
else:
    parameters=get_parameters(model,2e-5,0.95, 1e-4)
    optimizer = AdamW(parameters)

total_steps = len(train_data_loader)*EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    print(data_loader)
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() # 验证预测模式

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print('-'*10)
    train_acc,train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR,'best_model_state.ckpt'))
        best_accuracy = val_acc

test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)
print(f"test result is {test_acc.item()}")

def get_predictions(model, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values

y_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)
class_names = ['教育', '家居', '时尚', '时政', '科技', '房产', '财经']
print(classification_report(y_test, y_pred, target_names=[str(label) for label in class_names]))

