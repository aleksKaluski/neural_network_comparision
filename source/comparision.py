from source import multi_layer_perceptron as mlp

import source.prepare_data as prd
import source.dataset as dat
import source.multi_layer_perceptron as mlp
import source.table as tb
import source.recurent_neural_networks as rnn

def run_end_evaluate(model: str, dataset, embedding: str, learning_rate: float = 0.1, epochs: int = 100, units: int = 5):
    """
    Train and evaluate a single model. All possible intersections of embedding type and model type included.
    :param model: MLP, RNN, LSTM or GRU
    :param dataset: train-test dataset
    :param embedding: TDIDF, BOW or EBM (embedding)
    :return: accuracy (float) or a model
    """
    # standardise
    model = model.upper()
    embedding = embedding.upper()

    # Multi-layer perception with TD-IDF encodings
    if model == 'MLP' and embedding == 'TDIDF':
        X_train, X_test, Y_train, Y_test = dataset.get_encodings(tdidf=True)
        fnn = mlp.Feedforward_Model(X_train, Y_train, units=units)
        fnn.train(LR=learning_rate, epochs=epochs)
        return fnn.model.evaluate(X_test, Y_test, verbose=0)[1] # accuracy

    # Multi-layer perception with BOW encodings
    elif model == 'MLP' and embedding == 'BOW':
        X_train, X_test, Y_train, Y_test = dataset.get_encodings(tdidf=False)
        fnn = mlp.Feedforward_Model(X_train, Y_train, units=units)
        fnn.train(LR=learning_rate, epochs=epochs)
        return fnn.model.evaluate(X_test, Y_test, verbose=0)[1]  # accuracy

    # Recurrent Neural Network with embedding-based encodings
    elif model == 'RNN' and embedding == 'EBM':
        X_train, X_test, Y_train, Y_test = dataset.get_sequences(vocab_size=1000, maxlen=10)
        rec = rnn.Recurrent_Model(X=X_train, Y=Y_train, input_dim=1004, output_dim=1, units=10, rnn_type="RNN", two_layers=True)
        return rec.model.evaluate(X_test, Y_test, verbose=0)[1]

    # Long Short-Term Memory with embedding-based encodings
    elif model == 'LSTM' and embedding == 'EBM':
        X_train, X_test, Y_train, Y_test = dataset.get_sequences(vocab_size=1000, maxlen=10)

        # classification is binary, so output_dim=1 (YES/NO classification)
        rec = rnn.Recurrent_Model(X=X_train, Y=Y_train, input_dim=1004, output_dim=1, units=10, rnn_type="LSTM", two_layers=True)
        return rec.model.evaluate(X_test, Y_test, verbose=0)[1]

    # Gated recurrent units with embedding-based encodings
    elif model == 'GRU' and embedding == 'EBM':
        X_train, X_test, Y_train, Y_test = dataset.get_sequences(vocab_size=1000, maxlen=10)

        rec = rnn.Recurrent_Model(X=X_train, Y=Y_train, input_dim=1004, output_dim=1, units=10, rnn_type="GRU", two_layers=True)
        return rec.model.evaluate(X_test, Y_test, verbose=0)[1]

    else:
        raise ValueError(f'Invalid model-embedding combination. Model "{model}" is not compatible with embedding "{embedding}"')

def test_split_ratio(dataset,
                     model: str,
                     embedding: str,
                     learning_rate: float = 0.1,
                     epochs: int = 100,
                     units: int = 5):

    print(f"Initializing split test for {model}, with embedding {embedding}.")
    print(f"Params: {learning_rate}, {epochs}, {units}")
    result = []
    split = [0.1, 0.2, 0.3, 0.4, 0.5]

    for s in split:
        print('.', end='')
        dataset.split_dataset(test_size=s)
        a = run_end_evaluate(model, dataset, embedding, learning_rate, epochs, units)
        result.append((s, round(a, 4)))

    print(f"\nThe result is: {result}")
    return result

def test_vocab_size(df,
                     model: str,
                     embedding: str,
                     learning_rate: float = 0.1,
                     epochs: int = 100,
                     units: int = 5):

    print(f"Initializing vocab-size test for {model}, with embedding {embedding}.")
    print(f"Params: {learning_rate}, {epochs}, {units}")
    result = []
    features = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    for f in features:
        print('.', end='')
        dataset = dat.Text_Dataset(df, col_text="clean_text_str", col_label="sentiment", args={"max_features": f})
        dataset.split_dataset(test_size=0.2) # default test size
        a = run_end_evaluate(model, dataset, embedding, learning_rate, epochs, units)
        result.append((f, round(a, 4)))

    print(f"\nThe result is: {result}")
    return result



