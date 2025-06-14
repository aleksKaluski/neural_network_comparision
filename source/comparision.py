from source import multi_layer_perceptron as mlp

import source.prepare_data as prd
import source.dataset as dat
import source.multi_layer_perceptron as mlp
import source.table as tb



def test_split_ratio(dataset,
                     learning_rate: float = 0.1,
                     epochs: int = 100,
                     units: int = 5,
                     embeding: bool = False):
    print(f"Intializing test split with: lr = {learning_rate}, epochs = {epochs}, units = {units}.")
    if embeding:
        print("Embeding is TD-IDF.", end='')
    else:
        print("Embeding is BOW.", end='')

    result = []
    split = [0.1, 0.2, 0.3, 0.4, 0.5]

    for s in split:
        print('.', end='')
        dataset.split_dataset(test_size=s)
        X_train, X_test, Y_train, Y_test = dataset.get_encodings(tdidf=embeding)
        fnn = mlp.Feedforward_Model(X_train, Y_train, units=units)
        fnn.train(LR=learning_rate, epochs=epochs)

        acc = fnn.model.evaluate(X_test, Y_test, verbose=0)[1]
        result.append((s, round(acc, 4)))

    print(f"\nThe result is: {result}")
    return result


def test_vocab_size(df, learning_rate: float = 0.1, epochs: int = 100, units: int = 5, embeding: bool = False):

    print(f"Intializing test of vocab size with: lr = {learning_rate}, epochs = {epochs}, units = {units}.")
    if embeding:
        print("Embeding is TD-IDF.", end='')
    else:
        print("Embeding is BOW.", end='')

    features = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    result = []
    for f in features:
        print('.', end='')
        dataset = dat.Text_Dataset(df, col_text="clean_text_str", col_label="sentiment", args={"max_features": f})
        dataset.split_dataset(test_size=0.2) # deafult test size
        X_train, X_test, Y_train, Y_test = dataset.get_encodings(tdidf=embeding)
        fnn = mlp.Feedforward_Model(X_train, Y_train, units=units)
        fnn.train(LR=learning_rate, epochs=epochs)

        acc = fnn.model.evaluate(X_test, Y_test, verbose=0)[1]
        result.append((f, round(acc, 4)))

    print(f"\nThe result is: {result}")
    return result


