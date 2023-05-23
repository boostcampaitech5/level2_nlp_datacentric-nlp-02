import pandas as pd


class DataAugmentation():
    """
    config select DA에 명시된 Data Augmentation 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list

    def process(self, df):
        if self.select_list:
            aug_df = pd.DataFrame(columns=df.columns)

            for method_name in self.select_list:
                method = eval("self." + method_name)
                aug_df = pd.concat([aug_df, method(df)])

            df = pd.concat([df, aug_df])

        return df