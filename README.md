# Neural Network Model Comparison 

The aim of this project is to compare various models of neural networks to identify curial distinctions between them. Five combinations of models and encoding were examined.

| SHORTCUT | MODEL                      | ENCODING  |
|----------|----------------------------|-----------|
| MLP      | Multi layer perceptron     | BOW       |
| MLP      | Multi layer perceptron     | TD-IDF    |
| LSTM     | Long Short-Term Memory     | Embedding |
| GRU      | Gated recurrent units      | Embedding |
| RNN      | Recurrent neural networks  | Embedding |

## Comparison Criteria
- Models were compared according to a few different standards:

- Maximal accuracy achieved in 10 test runs

- Time complexity analysis

- Impact of the changing the test-split on model’s accuracy

- Impact of the changing of the vocab size on model’s accuracy

- Number of parameters to tune and train

## Installation 

Run the following command to install all required packages:

```bash
pip install pandas numpy spacy matplotlib scikit-learn tensorflow optuna tabulate ipython
```

After that, download the small English language model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Pipeline Overview

1. **Downloading the dataset from HuggingFace**  
   [https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment](https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment)

2. **Cleaning the records with spaCy**  
   Punctuation marks, non-alphabetical characters, and URLs were removed.

3. **Training various neural networks**
   - Training of MLP with TD-IDF encoding  
   - Training of MLP with BOW encoding  
   - Training of classical RNN with embedding-based encoding  
   - Training of classical LSTM with embedding-based encoding  
   - Training of classical GRU with embedding-based encoding  

4. **Uploading the data**  
   Data uploaded to `Table` class and `All_trials_table` class designed for keeping and plotting the data.

5. **Comparison of the models**

## A Few Remarks on the Project

1. **Data cleaning**  
   For cleaning the table from noise, I used spaCy. Please note that the cleaning is very soft (no stopwords removal) and only entirely useless cases are left out.

2. **Parameter tuning**  
   Tuning the parameters of each model was achieved with the help of `optuna` library. Broadly speaking, `optuna` is a state-of-the-art engine for searching the solution space. Since it is driven by algorithms, it is far more efficient than a common grid-search method. Each of the models had 10 runs with the same `optuna` algorithm.

3. **Data storage classes**  
   Two classes were prepared for keeping the data:
   - `Table`: stores the data of best runs as well as comparative studies regarding the vocab size and test-split.
   - `All_trials_table`: stores the data about each individual trial conducted by `optuna`.

## Summary of the results

##### Table with all the results:
| name      | epochs | accuracy | loss   | lr                 | units | avg_time  | split_accuracy                                                         | vocab_accuracy                                        | two_layers | batch_size |
|-----------|--------|----------|--------|--------------------|-------|-----------|------------------------------------------------------------------------|-------------------------------------------------------|------------|------------|
| rnn_emb   | 104    | 0.5720   | 0.7166 | 0.309436188253128  | 6     | 3.87679   | [(0.1, 0.57), (0.2, 0.5575), (0.3, 0.44), (0.4, 0.5725), (0.5, 0.554)] | [(1000, 0.5725), (2000, 0.4275), ..., (8000, 0.475)]  | 1          | 133        |
| lstm_emb  | 139    | 0.6310   | 3.3159 | 0.0139738427316055 | 7     | 5.757677  | [(0.1, 0.62), (0.2, 0.6425), ..., (0.5, 0.588)]                        | [(1000, 0.5475), (2000, 0.65), ..., (8000, 0.62)]     | 0          | 82         |
| gru_emb   | 105    | 0.6410   | 0.8266 | 0.227613209031116  | 6     | 12.026911 | [(0.1, 0.57), (0.2, 0.6125), ..., (0.5, 0.615)]                        | [(1000, 0.5725), (2000, 0.5825), ..., (8000, 0.5725)] | 0          | 17         |
| mlp_tdidf | 120    | 0.7000   | 0.5687 | 0.177334531431444  | 7     | 3.630928  | [(0.1, 0.72), (0.2, 0.68), ..., (0.5, 0.675)]                          | [(1000, 0.72), (2000, 0.7125), ..., (8000, 0.7125)]   | 0          | 0          |
| mlp_bow   | 122    | 0.6890   | 0.6339 | 0.0730366427984289 | 9     | 2.886153  | [(0.1, 0.69), (0.2, 0.6725), ..., (0.5, 0.695)]                        | [(1000, 0.68), (2000, 0.695), ..., (8000, 0.73)]      | 0          | 0          |

##### Best result was achieved by 
1) Multi layer perceptron with TD-IDF encoding type
2) Multi layer perceptron with BOW encoding type
3) Gated recurrent units with embedding encoding type

##### Plot of the impact of test split on the accuracy

![split_test_size](https://github.com/user-attachments/assets/0c49c09c-d936-48f4-a201-f2cd43a893ca)

##### Plot of the impact of the vocab size on the accuracy
![split_accuracy](https://github.com/user-attachments/assets/3cf85a11-cadf-42e0-ae11-7b42db74b588)

##### Plot of the time-complexity

![time_complexity](https://github.com/user-attachments/assets/b2a9132b-b1a8-48d2-ad88-6dbf645fecab)

##### Complexity 
RNN models (classical RNN, LSTM, GRU) have two more trainable parameters more then Multi-layer perceptron. Namely: batch size and an option to have two layers
