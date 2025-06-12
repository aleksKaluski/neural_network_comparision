import pandas as pd


class Table():

    def __init__(self):
        col_names = ['name', 'epochs', 'lr', 'units', 'avg_time', 'split_accuracy',  'vocab_accuracy']
        self.df = pd.DataFrame(columns=col_names)

    def add_record(self, name, epoch, lr, units, avg_time, split_accuracy, vocab_accuracy):
        assert isinstance(name, str), "Name must be a string"
        assert isinstance(epoch, int), "Epoch must be a integer"
        assert isinstance(lr, float), "Learning rate must be a float"
        assert isinstance(units, int), "Units must be a integer"
        assert isinstance(avg_time, float), "Average time must be a float"
        assert isinstance(split_accuracy, list), "split_accuracy must be a list"
        assert isinstance(vocab_accuracy, list), "vocab_accuracy must be a list"

        self.df.loc[len(self.df)] = [name, epoch, lr, units, avg_time, split_accuracy, vocab_accuracy]

    def toString(self):
        print("Info about the table: \n")
        print(self.df.info())
        print("\nThe table's head:")
        print(self.df.head())
