import pandas as pd
import re

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
    
    def zh_2_ko(self, df):
        """
        딕셔너리를 이용해서 자주 나오는 한자를 한국어로 변환 
        """
        zh_dict = {'美': '미국', '北': '북한', '中': '중국', '朴': '박근혜', '日': '일본', '靑': '청와대', '與': '여당', '英': '영국', 
                   '文': '문재인', '野': '야당', '獨': '독일', '伊': '이탈리아', '韓': '한국', '佛': '프랑스', '前': '전', '檢': '검찰', 
                   '軍': '군', '安': '안철수', '反': '반', '行': '행', '南': '남한', '亞': '아시아', '對': '대' , '硏': '연구원', 
                   '重': '중공업', '黃': '황교안' , '外': '외', '新': '새로운', '銀': '은행', '株': '주식', '展': '전시', '中企': '중소기업중앙회',
                   '車': '차', '親': '친', '孫': '손학규'}
        def is_zh(x): 
            l_ch = re.findall('[一-龥]+',x)
            if l_ch:
                for c in l_ch:
                    if c in zh_dict.keys():
                        x = re.sub(c,zh_dict[c],x)
                return x
            else:
                return x
        df['text'] = df['text'].apply(is_zh)
        
        return df