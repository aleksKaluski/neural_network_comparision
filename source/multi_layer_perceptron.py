import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
from itertools import product
import pandas as pd


class MLP(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=units, activation=tf.nn.sigmoid)
        self.output_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)


class Feedforward_Model:
    def __init__(self, X, Y, units=2):
        self.X = X
        self.Y = Y
        self.model = MLP(units)
        self.loss_acc = None

    # train the model
    def train(self, LR, epochs):
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        self.loss_acc = self.model.fit(self.X, self.Y, epochs=epochs, verbose=0)

    # plot loss and accuracy history
    def plot_loss_accuracy(self):
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle('Log Loss and Accuracy over epochs')

        # add_subplot(nrows, ncolumns, index)
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(self.loss_acc.history['loss'])
        ax.grid(True)
        ax.set(xlabel='Epochs', title='Log Loss')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(self.loss_acc.history['accuracy'])
        ax.grid(True)
        ax.set(xlabel='Epochs', title='Accuracy')


def find_best_mlp(X_train, Y_train, X_test, Y_test, n_trials=3):
    def train_mlp(trial):
        epochs = trial.suggest_int("epochs", 100, 200)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5)
        units = trial.suggest_int("units", 2, 10)

        fnn = Feedforward_Model(X_train, Y_train, units=units)
        fnn.train(LR=learning_rate, epochs=epochs)

        result = fnn.model.evaluate(X_test, Y_test, verbose=0)
        # loss function = result[0]
        # accuracy = result[1]
        return result[0], result[1]

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # loss, accuracy
        study_name='mlp_optimization'
    )
    study.optimize(train_mlp, n_trials=n_trials)

    ev_metric = study.trials_dataframe()
    ev_metric['time'] = ev_metric['duration'].dt.total_seconds()
    ev_metric.drop(columns=['datetime_start',
                            'datetime_complete',
                            'system_attrs_NSGAIISampler:generation',
                            'state',
                            'number',
                            'duration'],
                   inplace=True)
    ev_metric.rename(columns={'values_0': 'loss', 'values_1': 'accuracy'}, inplace=True)
    ev_metric.sort_values(by=['accuracy', 'loss'], ascending=False, inplace=True)
    ev_metric.reset_index(drop=True, inplace=True)
    ev_metric['time'].mean()




    best_params = ev_metric.iloc[0].to_dict()
    best_params['time'] = ev_metric['time'].mean()
    print(best_params)

#     print(f"""\nBest params:
#     accuracy: {best_params['accuracy'].round(4)}
#     loss function: {best_params['loss'].round(4)}
#     number of epochs: {best_params['params_epochs']}
#     learning rate: {best_params['params_learning_rate'].round(4)}
#     units: {best_params['params_units']}
# """)
    return best_params


