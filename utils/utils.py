import os
import torch
import numpy as np
import evaluate

from datetime import datetime, timezone, timedelta


def get_folder_name(CFG):
    """
    실험 결과를 기록하기 위한 고유 폴더명 생성

    ex) 이름001_테스트1, 이름002_테스트2
    """
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    use_data_str = "-".join([s.split(".")[0]  for s in CFG['select_data']])
    folder_name = f"{now.strftime('%d%H%M%S')}-{use_data_str}"
    save_path = f"./results/{folder_name}"
    CFG['save_path'] = save_path
    os.makedirs(save_path)

    return folder_name, save_path

def calc_accuracy(max_indices, Y):
    acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]

    return acc

def inference(model, data, tokenizer, DEVICE):
    model.eval()
    preds = []
    
    for idx, sample in data.iterrows():
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    return preds