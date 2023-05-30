import pandas as pd
import torch
import time

from koeda import SR, RI, RS, RD, EDA
from transformers import pipeline
from tqdm.auto import tqdm


class DataAugmentation():
    """
    config select DA에 명시된 Data Augmentation 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_DA, EDA_p):
        self.select_DA = select_DA
        self.EDA_p = EDA_p

    def process(self, df):
        use_file_list = df['track'].unique()

        for file_name in use_file_list:
            if self.select_DA[file_name]:
                aug_df = pd.DataFrame(columns=df.columns)

                for method_name in self.select_DA[file_name]:
                    print(f"{file_name}에 {method_name} 적용중 ===")
                    method = eval("self." + method_name)
                    aug_df = pd.concat([aug_df, method(df)])

                df = pd.concat([df, aug_df])
                print(f"적용 후 데이터 크기 >> {df.shape}")

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
    
    def EDA(self, df):
        """
        Easy Data Augmentation 기법 모두를 전체적으로 적용
        """
        p = self.EDA_p
        aug_df = df.copy()

        eda = EDA(
            morpheme_analyzer="Okt", alpha_sr=p, alpha_ri=p, alpha_rs=p, prob_rd=p
        )
        aug_df['text'] = aug_df['text'].apply(lambda x: eda(x))

        return aug_df

    
def back_translation_ko2en():
    df = pd.read_csv('data/train_spelling_v2_label_v1.csv')
    KETI = pipeline("translation", model="KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-ko2en",
                    src_lang="ko", tgt_lang="en", max_length=100)
    # KoJ = pipeline("translation", model="KoJLabs/nllb-finetuned-ko2en", src_lang="ko", tgt_lang="en", max_length=100)

    KETI_ko2en, KoJ_ko2en = [], []
    for idx, row in tqdm(df.iterrows()):
        KETI_ko2en.append(KETI(row['text'])[0]['translation_text'])
        # KoJ_ko2en.append(KoJ(row['text'])[0]['translation_text'][3:])
    
    df['KETI_ko2en'] = KETI_ko2en
    # df['KoJ_ko2en'] = KoJ_ko2en

    df.to_csv('data/train_spelling_v2_label_v1-KETI-ko2en.csv', index=False)

def back_translation_en2ko():
    df = pd.read_csv('data/train_spelling_v2_label_v1-KETI-ko2en.csv')
    KETI = pipeline("translation", model="KETI-AIR-Downstream/long-ke-t5-base-translation-aihub-en2ko",
                    src_lang="en", tgt_lang="ko", max_length=100)
    # KoJ = pipeline("translation", model="KoJLabs/nllb-finetuned-en2ko", src_lang="en", tgt_lang="ko", max_length=100)

    KETI_en2ko, KoJ_en2ko = [], []
    for idx, row in tqdm(df.iterrows()):
        KETI_en2ko.append(KETI(row['KETI_ko2en'])[0]['translation_text'])
        # KoJ_en2ko.append(KoJ(row['KoJ_ko2en'])[0]['translation_text'][3:])
    
    df['KETI_en2ko'] = KETI_en2ko
    # df['KoJ_en2ko'] = KoJ_en2ko

    df.to_csv('data/train_spelling_v2_label_v1-KETI-ko2en-en2ko.csv', index=False)


if __name__ == "__main__":
    back_translation_ko2en()
    time.sleep(2)
    back_translation_en2ko()