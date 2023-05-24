import os
import torch

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

def inference(model, dataloader, DEVICE):
    preds = []

    model.eval()
    for (token_ids, valid_length, segment_ids, label) in dataloader:
        token_ids = token_ids.long().to(DEVICE)
        segment_ids = segment_ids.long().to(DEVICE)

        out = model(token_ids, valid_length, segment_ids)

        _, max_indices = torch.max(out, 1)
        preds.extend(list(max_indices))

    return preds