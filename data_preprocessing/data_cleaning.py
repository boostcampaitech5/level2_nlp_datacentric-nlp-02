class DataCleaning():
    """
    config의 select DC에 명시된 Data Cleaning 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list

    def process(self, df, train=True):
        if self.select_list:
            for method_name in self.select_list:
                method = eval("self." + method_name)
                df = method(df)

        return df
    
