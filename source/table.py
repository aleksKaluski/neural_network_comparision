
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from IPython.display import display


class Table():
    """
    The class designed to keep and plot data about particular models.
    """
    def __init__(self):
        col_names = ['name', 'epochs', 'lr', 'units', 'avg_time', 'split_accuracy',  'vocab_accuracy', 'two_layers', 'batch_size']
        self.df = pd.DataFrame(columns=col_names)

    def add_record(self,
                   name,
                   epoch=0,
                   lr=0.0,
                   units=0,
                   avg_time=0.0,
                   two_layers=0,
                   batch_size=0,
                   split_accuracy=None,
                   vocab_accuracy=None):

        print(f"Adding a new record with params: name: {name}, epoch: {epoch}, lr: {lr}, units: {units}, average time: {avg_time}")

        if split_accuracy is None:
            split_accuracy = []
        if vocab_accuracy is None:
            vocab_accuracy = []


        self.df.loc[len(self.df)] = [name, epoch, lr, units, avg_time, split_accuracy, vocab_accuracy, two_layers, batch_size]

    def show(self, tabulate_view=True):
        """
        Print the contents of the table.
        :param tabulate_view: view for normal .py file (Should be False for notebook)
        """
        print("\nTable info:")
        self.df.info()

        print("\nRecords preview:")
        if tabulate_view:
            print(tabulate(self.df, headers='keys', tablefmt='github', showindex=False))
        else:
            display(self.df)

    def plot_split_accuracy(self):
        """
        Prints differt accuracies of various split methods.
        """
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        for idx, row in self.df.iterrows():
            data = row['split_accuracy']
            splits = [x for x, y in data]
            accuracies = [y for x, y in data]
            plt.plot(splits, accuracies, marker='o', label=row['name'])

        plt.title('Split Accuracy')
        plt.xlabel('Split')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)


    def plot_vocab_accuracy(self):
        """
        Prints differt accuracies of various sizes of vocab.
        """
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        for idx, row in self.df.iterrows():
            data = row['vocab_accuracy']
            vocab_sizes = [x for x, y in data]
            accuracies = [y for x, y in data]
            plt.plot(vocab_sizes, accuracies, marker='o', label=row['name'])

        plt.title('Vocab Size Accuracy')
        plt.xlabel('Vocab Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)


    def plot_time_complexity(self):
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)


        for idx, row in self.df.iterrows():
            time = row['avg_time']
            epochs = row['epochs']
            plt.plot(time, epochs, marker='o', label=row['name'])

        plt.title('Time complexity')
        plt.xlabel('Time')
        plt.ylabel('Number of epochs')
        plt.legend()
        plt.grid(True)


class All_trials_table():
    """
    Class for keeping data about time complexity.
    """
    def __init__(self):
        col_names = ['loss', 'accuracy', 'params_epochs', 'params_learning_rate', 'params_units', 'time', 'trial_time', 'name']
        self.df = pd.DataFrame(columns=col_names)


    def add_record(self,
                   df):

        print(f"Adding a new record with params.")
        self.df =pd.concat([self.df, df], ignore_index=True)


    def show(self, tabulate_view=True):
        """
        Print the contents of the table.
        :param tabulate_view: view for normal .py file (Should be False for notebook)
        """
        print("\nTable info:")
        self.df.info()

        print("\nRecords preview:")
        if tabulate_view:
            print(tabulate(self.df, headers='keys', tablefmt='github', showindex=False))
        else:
            display(self.df)

    import matplotlib.pyplot as plt

    def plot_time_complexity(self):
        plt.figure(figsize=(10, 5))
        p = self.df.copy()

        p.sort_values(by=['trial_time', 'params_epochs'], ascending=False, inplace=True)
        for name in self.df['name'].unique():
            subset = self.df[self.df['name'] == name]
            plt.scatter(subset['trial_time'], subset['params_epochs'], label=name)


        plt.title('Time Complexity by Model')
        plt.xlabel('Trial Time')
        plt.ylabel('Number of Epochs')
        plt.legend(title='Model Name')
        plt.grid(True)
        plt.tight_layout()
        plt.show()




