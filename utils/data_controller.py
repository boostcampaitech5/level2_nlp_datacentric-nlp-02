import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from data_preprocessing.data_cleaning import DataCleaning


class BERTDataset(Dataset):
    def __init__(self, dataset, transform):
        texts = dataset['input_text'].tolist()
        targets = dataset['target'].tolist()

        self.sentences = [transform([i]) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
def get_train_dataset(CFG, SEED):
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for file_name in CFG['select_data']:
        view_df = pd.read_csv(f'data/{file_name}.csv')
        
        DC = DataCleaning(CFG['select_DC'][file_name])
        view_df_after_DC = DC.process(view_df)
        view_df_after_DC['track'] = file_name

        if file_name == "train":
            view_df_after_DC, val_df = train_test_split(view_df_after_DC, train_size=0.7, random_state=SEED)

        train_df = pd.concat([train_df, view_df_after_DC], axis=0)
    
    train_df.drop_duplicates(subset=['input_text', 'target']) # input_text만 동일하고 target 다른 경우 있음 -> 나중에 확인

    return train_df, val_df

def get_test_dataset():
    df = pd.read_csv('data/test.csv')

    return df