import pandas as pd

from collections import Counter


class DataCleaning():
    """
    config의 select DC에 명시된 Data Cleaning 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list
        self.label2num = {
            label: i for i, label in enumerate(['정치', '경제', '사회', '생활문화', '세계', 'IT과학', '스포츠'])
        }

    def process(self, df):
        if self.select_list:
            for method_name in self.select_list:
                method = eval("self." + method_name)
                df = method(df)

        return df
    
    def add_news_category(self, df):
        """
        predefined_news_category를 문장 뒤에 추가
        """
        df['text'] = df['text'] + " [SEP] " + df['predefined_news_category']

        return df
    
    def label_all_predefind(self, df):
        """
        label_text를 target으로 사용하지 않고 predefined_news_category를 target으로 사용하기
        """
        df.drop_duplicates(subset=['ID', 'text'], inplace=True)
        df['target'] = df['predefined_news_category'].apply(lambda x: self.label2num[x])

        return df
    
    def all_predefind_for_label_error(self, df):
        """
        label error 상태인 데이터의 target을 모두 predefind 값으로 바꾸기
        """
        label_error_df = df[df.duplicated(['ID', 'text'], keep=False)]
        df.drop(label_error_df.index, axis=0, inplace=True)
        
        label_error_df['target'] = label_error_df['predefined_news_category'].apply(lambda x: self.label2num[x])
        label_error_df.drop_duplicates(subset=['ID'], inplace=True)

        return pd.concat([df, label_error_df], axis=0)
    
    def mix_mix(self, df):
        """
        shuffle을 못 쓰니 직접 dataframe을 뒤섞어 주는 메소드
        """
        df = df.sample(frac=1, random_state=456, axis=0)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def add_label_token(self, df):
        """
        text 앞에 target 정보를 입력하기
        """
        label2num = {
            k: v for k, v in enumerate(['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치'])
        }

        df['text'] = df['target'].apply(lambda x: label2num[x]) + " [SEP] " + df['text']

        return df
    
    def add_most_word_5(self, df):
        """
        target별 빈도수 기준 상위 5개 단어 붙이기
        """
        label2top5 = dict()
        for target in range(7):
            view_df = df[df['target'] == target]

            arr = []
            for row in view_df['text'].to_list():
                arr.extend(row.split())

            counter = Counter(arr)
            label2top5[target] = [most for most, count in counter.most_common(5)]
        
        df['text'] = df['target'].apply(lambda x: " ".join(label2top5[x])) + " [SEP] " + df['text']

        return df