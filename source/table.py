from os.path import split

import pandas as pd
from tabulate import tabulate

class Table():
    def __init__(self):
        col_names = ['name', 'epochs', 'lr', 'units', 'avg_time', 'split_accuracy',  'vocab_accuracy']
        self.df = pd.DataFrame(columns=col_names)

    def add_record(self, name, epoch, lr, units, avg_time, split_accuracy, vocab_accuracy):
        print(f"Adding a new record with params: name: {name}, epoch: {epoch}, lr: {lr}, units: {units}, average time: {avg_time}")
        assert isinstance(name, str), "Name must be a string"
        assert isinstance(epoch, int) or isinstance(epoch, float), "Epoch must be a integeror float"
        assert isinstance(lr, float), "Learning rate must be a float"
        assert isinstance(units, int) or isinstance(units, float), "Units must be a integer"
        assert isinstance(avg_time, float), "Average time must be a float"
        assert isinstance(split_accuracy, list), "split_accuracy must be a list"
        assert isinstance(vocab_accuracy, list), "vocab_accuracy must be a list"

        self.df.loc[len(self.df)] = [name, epoch, lr, units, avg_time, split_accuracy, vocab_accuracy]

    def show(self, tabulate_view=True):
        print("\nTable info:")
        print(self.df.info())
        print("\nRecords preview:")

        if tabulate_view:
            print(tabulate(self.df, headers='keys', tablefmt='github', showindex=False))
        else:
            print(self.df.head())

    def print_split_accuracy(self):
        name_record = []
        for row in self.df.iterrows():
            name_record.append((row["name"], row["split_accuracy"]))
        print(name_record)
