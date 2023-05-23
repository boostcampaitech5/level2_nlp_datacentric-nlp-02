import numpy as np
import gluonnlp as nlp
import torch
import yaml
import wandb

from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from shutil import copyfile
from sklearn.metrics import confusion_matrix

from models import models
from utils import data_controller, utils

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
# 넘파이 랜덤시드 고정
np.random.seed(SEED)
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
    train_df = data_controller.get_train_dataset(CFG)
    train, val = train_test_split(train_df, train_size=0.7, random_state=SEED)

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
    val_pred_y = []
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

            train_acc += utils.calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

            wandb.log({"train loss": loss,
                       "train accuracy": train_acc / (batch_id+1)})
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

        with torch.no_grad():
            model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                token_ids = token_ids.long().to(DEVICE)
                segment_ids = segment_ids.long().to(DEVICE)
                label = label.long().to(DEVICE)

                out = model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)

                val_pred_y.append(out)
                val_acc += utils.calc_accuracy(out, label)

                wandb.log({"val loss": loss,
                           "val accuracy": val_acc / (batch_id+1)})
            print("epoch {} val acc {}".format(e+1, val_acc / (batch_id+1)))

    ### Evaluate Model ###
    test_df = data_controller.get_test_dataset()
    test_df['target'] = [0]*len(test_df)
    data_test = data_controller.BERTDataset(test_df, transform)
    eval_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    preds = []
    test_acc = 0.0
    for batch_id, (token_ids, valid_length, segment_ids, _) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        token_ids = token_ids.long().to(DEVICE)
        segment_ids = segment_ids.long().to(DEVICE)
        
        out = model(token_ids, valid_length, segment_ids)

        max_vals, max_indices = torch.max(out, 1)
        preds.extend(list(max_indices))

    preds = [int(p) for p in preds]
    test_df['target'] = preds
    test_df.to_csv(f"{save_path}/{folder_name}_submit.csv", index=False)