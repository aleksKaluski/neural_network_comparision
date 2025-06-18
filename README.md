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


