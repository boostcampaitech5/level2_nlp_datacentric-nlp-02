import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from data_preprocessing.data_cleaning import DataCleaning


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)
    
def get_train_dataset(CFG, SEED):
    """
    config에 명시한 select data를 
    """
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    if CFG['option']['stack']['trigger']:
        file_name = CFG['select_data'][0]
        print("단일 데이터셋을 여러 번 쌓는 실험")
        for _ in range(CFG['option']['stack']['stack_num']):
            view_df = pd.read_csv(f"data/{file_name}.csv")


        if idx == 0:
            view_df_after_DC, val_df = train_test_split(view_df_after_DC, test_size=0.3, random_state=SEED)
        
        if 'bt' in file_name:
            
            train_bt = view_df_after_DC.copy()
            train_bt['text'] = train_bt['bt']
            view_df_after_DC = pd.concat([view_df_after_DC,train_bt], axis=0)
            view_df_after_DC = view_df_after_DC[['ID','text','target','url','date','track']]

            DC = DataCleaning(CFG['select_DC'][file_name])
            view_df_after_DC = DC.process(view_df)
            view_df_after_DC['track'] = file_name


            view_df_after_DC, _val_df = train_test_split(view_df_after_DC, test_size=0.3, random_state=SEED)

            train_df = pd.concat([train_df, view_df_after_DC], axis=0)
            val_df = pd.concat([val_df, _val_df], axis=0)
    else:
        print("다양한 데이터셋을 사용하는 실험")
        for idx, file_name in enumerate(CFG['select_data']):
            view_df = pd.read_csv(f'data/{file_name}.csv')
            
            DC = DataCleaning(CFG['select_DC'][file_name])
            view_df_after_DC = DC.process(view_df)
            view_df_after_DC['track'] = file_name

            if idx == 0:
                view_df_after_DC, val_df = train_test_split(view_df_after_DC, test_size=0.3, random_state=SEED)

            train_df = pd.concat([train_df, view_df_after_DC], axis=0)
    
        train_df.drop_duplicates(subset=['text', 'target'], inplace=True)
    
    # LOG: 데이터 크기 확인
    print(f"train_df shape >> {train_df.shape}\t\tval_df shape >> {val_df.shape}")

    return train_df[['text', 'target', 'track']], val_df[['text', 'target']]

def get_test_dataset():
    """
    test.csv 파일을 불러오는 메소드
    """
    df = pd.read_csv('data/test.csv')

    return df