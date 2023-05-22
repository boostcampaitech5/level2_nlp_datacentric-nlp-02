import os


def get_folder_name(CFG):
    """
    실험 결과를 기록하기 위한 고유 폴더명 생성

    ex) 이름001_테스트1, 이름002_테스트2
    """
    folder_name = f"{CFG['code']}_{CFG['summary']}"
    save_path = f"./results/{folder_name}"
    CFG['save_path'] = save_path
    os.makedirs(save_path)

    return folder_name, save_path

def create_new_data():
    """
    annotations을 분리한 새로운 데이터셋 만드는 메소드
    """
    pass


if __name__ == "__main__":
    pass