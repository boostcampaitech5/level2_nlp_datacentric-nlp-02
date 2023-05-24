import pandas as pd


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