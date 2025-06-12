import matplotlib.pyplot as plt
import tensorflow as tf

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