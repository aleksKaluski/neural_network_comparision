import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import product
from pandas import DataFrame


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

        # output layer
        self.output_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name='output_layer')

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
        self.model = Rec_Unit(input_dim, output_dim, units, rnn_type, two_layers)

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