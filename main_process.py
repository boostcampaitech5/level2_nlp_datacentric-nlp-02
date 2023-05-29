import os
import random
import numpy as np
import torch
import yaml
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate

from shutil import copyfile
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from data import tokenization_kobert
from utils import data_controller, utils
from data_preprocessing.data_augmentation import DataAugmentation

import warnings
warnings.filterwarnings('ignore')


f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return f1.compute(predictions=predictions, references=labels, average='macro')


### Setting parameters ###
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# SEED 고정
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
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
    model_name = 'monologg/kobert'
    tokenizer = tokenization_kobert.KoBertTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    ### Define Dataset ###
    train, val = data_controller.get_train_dataset(CFG, SEED)
    # data augmentation
    DA = DataAugmentation(CFG['select_DA'])
    train = DA.process(train)

    data_train = data_controller.BERTDataset(train, tokenizer)
    data_val = data_controller.BERTDataset(val, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ### Train Model ###
    training_args = TrainingArguments(
    output_dir=save_path,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy='steps',
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    learning_rate= 2e-05,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    seed=SEED,
    report_to="wandb"
    )   

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    trainer.train()

    ### val datasest 예측 후 결과 저장 ###
    pred_vals = utils.inference(model, val, tokenizer, DEVICE)
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
    pred_tests = utils.inference(model, test, tokenizer, DEVICE)
    
    test['target'] = pred_tests
    test.to_csv(f"{save_path}/{folder_name}_submit.csv", index=False)