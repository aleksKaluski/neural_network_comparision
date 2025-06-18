import matplotlib.pyplot as plt
import tensorflow as tf
import optuna



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


def find_best_mlp(X_train, Y_train, X_test, Y_test, n_trials:int = 3, encoding: str = 'BOW'):
    """
    Find the best MLP model using optuna (external library for tuning parameters - more efficient than grid search).
    The data must in the form of TD-IDF or BOW.
    :param n_trials: number of trials for optuna
    :return: a dictionary with the best MLP model configuration
    """
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

    # the last value is the best one
    ev_metric.sort_values(by=['accuracy', 'loss'], ascending=True, inplace=True)
    ev_metric.reset_index(drop=True, inplace=True)
    ev_metric['trial_time'] = ev_metric['time']
    ev_metric['time'].mean()

    # set units as int
    ev_metric['params_units'] = ev_metric['params_units'].astype(int)
    ev_metric['name'] = 'MLP_' + encoding

    return ev_metric


