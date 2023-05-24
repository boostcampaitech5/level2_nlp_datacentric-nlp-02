import numpy as np
import gluonnlp as nlp
import torch
import yaml
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import random

from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from shutil import copyfile
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

from models import models
from utils import data_controller, utils
from data_preprocessing.data_augmentation import DataAugmentation

import warnings
warnings.filterwarnings('ignore')


### Setting parameters ###
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
DEVICE = torch.device('cuda')
SEED = 42
# 파이토치의 랜덤시드 고정
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED) # 넘파이 랜덤시드 고정
random.seed(SEED)
# config 파일 불러오기
with open('./use_config.yaml') as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    # 실험 결과 파일 생성 및 폴더명 가져오기
    folder_name, save_path = utils.get_folder_name(CFG)
    copyfile('use_config.yaml', f"{save_path}/config.yaml")
    # wandb 설정
    wandb.init(name=folder_name, project=CFG['wandb']['project'], 
               config=CFG, entity=CFG['wandb']['id'], dir=save_path)

    ### Load Tokenizer and Model ###
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len, pad=True, pair=False)

    ### Define Dataset ###
    train, val = data_controller.get_train_dataset(CFG, SEED)
    # data augmentation
    DA = DataAugmentation(CFG['select_DA'])
    train = DA.process(train)

    data_train = data_controller.BERTDataset(train, transform)
    data_val = data_controller.BERTDataset(val, transform)

    train_dataloader = DataLoader(data_train, batch_size=batch_size)
    val_dataloader = DataLoader(data_val, batch_size=batch_size)

    ### Define Model ###
    model = models.BERTClassifier(bertmodel, dr_rate=0.5).to(DEVICE)

    ### Define Optimizer and Scheduler ###
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    ### Train Model ###
    for e in range(num_epochs):
        train_acc = 0.0
        val_acc = 0.0
        wandb.log({"epoch": e})

        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            token_ids = token_ids.long().to(DEVICE)
            segment_ids = segment_ids.long().to(DEVICE)
            label = label.long().to(DEVICE)
            
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            max_vals, max_indices = torch.max(out, 1)
            train_acc += utils.calc_accuracy(max_indices, label)

            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

            wandb.log({"train loss": loss,
                       "train accuracy": train_acc / (batch_id+1)})
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                token_ids = token_ids.long().to(DEVICE)
                segment_ids = segment_ids.long().to(DEVICE)
                label = label.long().to(DEVICE)

                out = model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)

                max_vals, max_indices = torch.max(out, 1)
                val_acc += utils.calc_accuracy(max_indices, label)

                wandb.log({"val loss": loss,
                           "val accuracy": val_acc / (batch_id+1),
                           "val f1 macro": f1_score(label.cpu(), max_indices.cpu(), average='macro')})
            print("epoch {} val acc {}".format(e+1, val_acc / (batch_id+1)))

    ### val datasest 예측 후 결과 저장 ###
    pred_vals = [int(p) for p in utils.inference(model, val_dataloader, DEVICE)]
    val['pred_y'] = pred_vals
    val.to_csv(f"{save_path}/{folder_name}_val.csv", index=False)
    # confusion matrix
    classes = ['Politics', 'Economy', 'Society', 'Culture', 'World', 'IT/Science', 'Sport']
    CM = confusion_matrix(val['target'], val['pred_y'])
    plt.figure(figsize=(15, 10))
    # 카운트
    sns.heatmap(CM, annot=True, fmt="d", linewidths = 0.01, cmap='jet')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{save_path}/{folder_name}_CM_count.png')
    # 퍼센트
    plt.figure(figsize=(15, 10))
    CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
    sns.heatmap(CM_norm, annot=True, fmt=".2f", linewidths = 0.01,
                cmap='jet', vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{save_path}/{folder_name}_CM_per.png')
    # wandb confusion matrix
    wandb.log({
        "CM": wandb.plot.confusion_matrix(
            y_true=val['target'].values, preds=val['pred_y'].values,
            class_names=classes
        )
    })

    ### Evaluate Model ###
    test = data_controller.get_test_dataset()
    test['target'] = [0] * len(test)
    data_test = data_controller.BERTDataset(test, transform)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    
    pred_tests = [int(p) for p in utils.inference(model, test_dataloader, DEVICE)]
    test['target'] = pred_tests
    test.to_csv(f"{save_path}/{folder_name}_submit.csv", index=False)