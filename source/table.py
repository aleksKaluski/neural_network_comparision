import pandas as pd


class Table:
    def __init__(self, args={}):
        self.df = pd.DataFrame()
        self.args = args
