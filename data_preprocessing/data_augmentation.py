import pandas as pd

from koeda import SR, RI, RS, RD


class DataAugmentation():
    """
    config select DA에 명시된 Data Augmentation 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_DA):
        self.select_DA = select_DA

    def process(self, df):
        use_file_list = df['track'].unique()

        for file_name in use_file_list:
            if self.select_DA[file_name]:
                aug_df = pd.DataFrame(columns=df.columns)

                for method_name in self.select_DA[file_name]:
                    method = eval("self." + method_name)
                    aug_df = pd.concat([aug_df, method(df)])

                df = pd.concat([df, aug_df])

        return df
    
    def random_deletion(self, df):
        """
        Easy Data Augmentation 기법 중 랜덤으로 데이터 삭제
        """
        aug_df = df.copy()

        func = RD("Okt")
        aug_df['text'] = func(aug_df['text'].to_list(), 0.1)

        return aug_df
    
    def random_swap(self, df):
        """
        Easy Data Augmentation 기법 중 랜덤으로 단어 위치 바꿈
        """
        aug_df = df.copy()

        func = RS("Okt")
        aug_df['text'] = func(aug_df['text'].to_list(), 0.1)

        return aug_df