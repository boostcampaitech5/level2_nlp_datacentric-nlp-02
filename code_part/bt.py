import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

import googletrans
import re

from collections import Counter
from tqdm import tqdm
from hanspell import spell_checker

# 경고메세지 끄기
# warnings.filterwarnings(action='ignore')


train_df=pd.read_csv('/opt/ml/level2_nlp_datacentric-nlp-02/data/train_spelling_v2_label_v1.csv')
# /opt/ml/level2_nlp_datacentric-nlp-02/data/train.csv


trans = googletrans.Translator()
ch_dict = {'美': '미국', '北': '북한', '中': '중국', '朴': '박근혜', '日': '일본', '靑': '청와대', '與': '여당', '英': '영국', '文': '문재인', '野': '야당', '獨': '독일', '伊': '이탈리아', '韓': '한국', '佛': '프랑스', '前': '전', '檢': '검찰', '軍': '군', '安': '안철수', '反': '반', '行': '행', '南': '남한', '亞': '아시아', '對': '대' , '硏': '연구원', '重': '중공업', '黃': '황교안' , '外': '외', '新': '새로운', '銀': '은행', '株': '주식', '展': '전시', '中企': '중소기업중앙회','車': '차', '親': '친', '孫': '손학규'}

def is_ch(x):
    l_ch = re.findall('[一-龥]+',x)
    if l_ch:
        for c in l_ch:

            if c in ch_dict.keys():
                x = re.sub(c,ch_dict[c],x)
        return x
    else:
        return x
def bt(x):
    x = trans.translate(x,dest='en')
    x = trans.translate(x.text,dest='ko')
    return x.text

a = []
for i in range(len(train_df)//5000):
    a.append((i*5000,(i+1)*5000))
a.append(((i+1)*5000,len(train_df)))
print('-----------------start--------------')

for k,(i,j) in enumerate(a):
    if k>= 4:
        print(k)
        tqdm.pandas()
        train_bt = train_df.loc[i:j].copy()
        train_bt['text'] = train_bt['text'].apply(is_ch)
        train_bt['text'] = train_bt['text'].progress_apply(bt)

        name = '/opt/ml/level2_nlp_datacentric-nlp-02/data/bt/train_btg_'+str(k)+'.csv'
        train_bt.to_csv(name,

                        sep=',',

                        na_rep='NaN') # do not write index