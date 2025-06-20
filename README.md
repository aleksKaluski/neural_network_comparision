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
In case of any problems with compatibility environment, install the exact versions of this project provided in the `requirements.txt` file

```bash
pip install -r requirements.txt
```

After that, download the small English language model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Info regarding the dataset
Set of English tweets stored with table and annotated with sentiment label. 
- 0 — negative sentiment
- 1 — positive sentiment

Structure of the dataset:
| ID | Tweet                                                                                     |
|----|-------------------------------------------------------------------------------------------|
| 1  | @tonigirl14 love you toooooo!! TG LOL Gngb                                                |
| 0  | @jun6lee I told myself: Don't click on this link. But I just did. Booohooo               |

Full dataset contained 179995 instances, but for this comparison I decided to slice 2000 tweets, since parameter tuning and training time takes a lot of time. 
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

## Conclusion 
The best accuracy was achieved Multi-layer Perceptron with TD-IDF encoding type for 2000 tweets. Bellow is a short comparison of three best models. 

| Model     | Epochs | Accuracy | Loss  | Learning rate | Units | Average time of training | Has two layers? | Batch Size | Best split-test | Best vocab size |
|-----------|--------|----------|-------|----------------|--------|---------------------------|------------------|-------------|------------------|------------------|
| mlp_tdidf | 120    | 0.700    | 0.569 | 0.177          | 7      | 3.631                     | NO               | 0           | 0.1              | 5000             |
| mlp_bow   | 122    | 0.689    | 0.634 | 0.073          | 9      | 2.886                     | NO               | 0           | 0.2              | 1000             |
| gru_emb   | 105    | 0.641    | 0.827 | 0.228          | 6      | 12.027                    | NO               | 17          | 0.2              | 3000             |
| lstm_emb  | 139    | 0.631    | 3.316 | 0.014          | 7      | 5.758                     | NO               | 82          | 0.1              | 7000             |
| rnn_emb   | 104    | 0.572    | 0.717 | 0.309          | 6      | 3.877                     | YES              | 133         | 0.1              | 1000             |


It is clear that word-frequency based methods gave better results on such a small dataset (2000 tweets). The quenstion is: why? On a small datasets, with simple structure, as presented, there is much less information to learn from. RNNs (botch classical as well as LSTM or GRU) have far more complex structure. Contrastingly, a simple structure of MLP takes both less time to learn and can easier adjust to small, undemanding datasets. We should recall two important things:
1. The models had just 10 runs to tune
2. The dataset was very small (2000 short tweets)
For such a small amount of data, encoding techniques are usually much better than the ones with embeddings. Another interesting feature is low time complexity which is almost entirely independent from the number of epochs and type of the model. It might be the special feature of using tweets, since they are too short to provide any meaningful difference in terms of RNN advantages.

What happens when we make the dataset bigger? For instance: if we take a 10000 rows intead of 2000. Still, it is a relatively small amout of data, so the domination of MLP persists. However it is sufficiently large to provide a change in the table. 
| Model     | Epochs | Accuracy | Loss  | Learning rate | Units | Average time of training | Has two layers? | Batch Size | Best split-test | Best vocab size |
|-----------|--------|----------|-------|----------------|--------|---------------------------|------------------|-------------|------------------|------------------|
| mlp_bow   | 146    | 0.733    | 0.549 | 0.024          | 9      | 16.647                    | NO               | 0           | 0.1              | 5000             |
| mlp_tdidf | 200    | 0.730    | 0.533 | 0.045          | 7      | 29.877                    | NO               | 0           | 0.2              | 4000             |
| lstm_emb  | 193    | 0.673    | 0.662 | 0.175          | 2      | 15.924                    | YES              | 231         | 0.4              | 5000             |
| gru_emb   | 200    | 0.671    | 0.843 | 0.043          | 4      | 18.012                    | YES              | 235         | 0.1              | 2000             |
| rnn_emb   | 100    | 0.597    | 0.709 | 0.059          | 5      | 6.636                     | NO               | 116         | 0.4              | 7000             |

Now the LSTM model is the third one. What is even more interesting is a significant difference in the training time. 



![time_10000.png](photos\time_10000.png)