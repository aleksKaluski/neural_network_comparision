import matplotlib.pyplot as plt
import tensorflow as tf
import optuna

class Rec_Unit(tf.keras.Model):
    '''
    Recurernt model arguments:
    - input_dim: <int> (size of the vocabulary)
    - output_dim: <int> (dimension of the embedding vector)
    - units: <int> (number of units per layer)
    - rnn_type: <str> (type of recurrent unit: RNN, LSTM, or GRU)
    - two_layers: <bool> (True: two recurrent layers or False: one recurrent layer)
    Congiguration (Layers):
    - Embedding Layer
    - RNN|LSTM|GRU Layer 1
    - RNN|LSTM|GRU Layer 2 (optional)
    - Output Layer
    '''
    def __init__(self, input_dim, output_dim, units, rnn_type, two_layers):
        super().__init__()
        # select type of layer
        RNN_layer = {'RNN': tf.keras.layers.SimpleRNN,
                     'LSTM': tf.keras.layers.LSTM,
                     'GRU': tf.keras.layers.GRU}.get(rnn_type)

        # embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, name='embedding_layer')

        # first RNN layer
        self.layer1 = RNN_layer(units=units, activation=tf.nn.tanh, return_sequences=two_layers, name='rnn_layer1')

        # optional second RNN layer
        self.layer2 = None
        if two_layers:
            self.layer2 = RNN_layer(units=units, activation=tf.nn.tanh, name='rnn_layer2')

        # sigmoid output layer, since the classification is binary
        self.output_layer = tf.keras.layers.Dense(units=output_dim, activation=tf.nn.sigmoid, name='output_layer')

    def build(self, input_shape):
        '''
        Initializes trainable weights for each layer.
        '''
        self.embedding_layer.build(input_shape)
        input_shape = self.embedding_layer.compute_output_shape(input_shape)
        self.layer1.build(input_shape)
        input_shape = self.layer1.compute_output_shape(input_shape)
        if self.layer2:
            self.layer2.build(input_shape)
            input_shape = self.layer2.compute_output_shape(input_shape)
        self.output_layer.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.layer1(x)
        if self.layer2:
            x = self.layer2(x)
        return self.output_layer(x)


class Recurrent_Model:
    def __init__(self, X, Y, input_dim, output_dim, units, rnn_type, two_layers):
        self.X = X
        self.Y = Y

        # output dim must be 1 (binary analysis), so I leave it here
        self.output_dim = output_dim

        # input dim = vocab size
        self.input_dim = input_dim

        # type of the RNN
        self.rnn_type = rnn_type

        self.model = Rec_Unit(input_dim, output_dim, units, rnn_type, two_layers)

        # model for binary sentiment analysis
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    # train the model
    def train(self, LR, epochs, batch_size=32, verbose=0, validation_split=None):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        self.loss_acc = self.model.fit(self.X, self.Y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    # plot loss and accuracy history
    def plot_loss_accuracy(self):
        fig = plt.figure(figsize=(8, 4))
        fig.suptitle('Log Loss and Accuracy over Epochs')
        labels = ['Training', 'Validation']

        # get training history to plot
        loss = self.loss_acc.history.get('loss')
        val_loss = self.loss_acc.history.get('val_loss')
        accuracy = self.loss_acc.history.get('accuracy')
        val_accuracy = self.loss_acc.history.get('val_accuracy')

        # add_subplot(nrows, ncolumns, index)
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(loss, label=labels[0])
        if val_loss:
            ax.plot(val_loss, label=labels[1])
        ax.grid(True)
        ax.set(xlabel='Epochs', title='Log Loss')
        ax.legend(loc='upper right')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(accuracy, label=labels[0])
        if val_accuracy:
            ax.plot(val_accuracy, label=labels[1])
        ax.grid(True)
        ax.set(xlabel='Epochs', title='Accuracy')
        ax.legend(loc='lower right')


def find_best_rnn(X_train, Y_train, X_test, Y_test, n_trials=3, input_dim=1004, output_dim=1, rnn_type="RNN"):
    def train_and_evaluate_rnn(trial):

        epochs = trial.suggest_int("epochs", 100, 200)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5)
        units = trial.suggest_int("units", 2, 10)
        two_layers = trial.suggest_categorical("two_layers", [True, False])
        batch_size = trial.suggest_int("batch_size", 5, 256)

        rnn = Recurrent_Model(X=X_train,
                            Y=Y_train,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            units=units,
                            rnn_type=rnn_type,
                            two_layers=two_layers)


        rnn.train(LR=learning_rate,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=1,
                  validation_split=0.2)

        result = rnn.model.evaluate(X_test, Y_test, verbose=0)
        # loss function = result[0]
        # accuracy = result[1]

        return result[0], result[1]

    study = optuna.create_study(
        directions=["minimize", "maximize"],  # minimize loss, maximize accuracy
        study_name='mlp_optimization'
    )
    study.optimize(train_and_evaluate_rnn, n_trials=n_trials)
    ev_metric = study.trials_dataframe()

    # format time
    ev_metric['time'] = ev_metric['duration'].dt.total_seconds()

    # drop unnecessary columns
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

    # compute avg. time of all runs
    ev_metric['trial_time'] = ev_metric['time']
    ev_metric['time'].mean()
    ev_metric['name'] = str(rnn_type) + '_EBM'

    return ev_metric
