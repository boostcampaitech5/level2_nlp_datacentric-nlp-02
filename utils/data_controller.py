import pandas as pd
import numpy as np

from torch.utils.data import Dataset

from data_preprocessing.data_cleaning import DataCleaning
from data_preprocessing.data_augmentation import DataAugmentation


class BERTDataset(Dataset):
    def __init__(self, dataset, transform):
        texts = dataset['input_text'].tolist()
        targets = dataset['target'].tolist()

        self.sentences = [transform(i) for i in texts]
        self.labels = [np.int32(i) for i in targets]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
def get_train_dataset(CFG):
    df = pd.DataFrame()

    for file_name in CFG['select_data']:
        view_df = pd.read_csv(f'data/{file_name}.csv')
        
        DC = DataCleaning(CFG['select_DC'][file_name])
        view_df_after_DC = DC.process(view_df)
        
        DA = DataAugmentation(CFG['select_DA'][file_name])
        view_df_after_DC_and_DA = DC.process(view_df_after_DC)

        df = pd.concat([df, view_df_after_DC_and_DA], axis=0)
    
    return df

def get_test_dataset():
    df = pd.read_csv('data/test.csv')

    return df