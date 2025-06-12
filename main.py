import pandas as pd
import numpy as np
import spacy
import os
from pathlib import Path

from source import prepare_data
from source import dataset as dat
from source import multi_layer_perceptron as mlp

"""
1) Preprocessing
"""
# load models and dataset
df = pd.read_csv("hf://datasets/gxb912/large-twitter-tweets-sentiment/train.csv", nrows=100)
nlp = spacy.load("en_core_web_sm")

# prepare dataset
df = prepare_data.prepare_df(df, nlp)

# create dataset out of cleaned columns
dataset = dat.Text_Dataset(df, col_text="clean_text_str", col_label="sentiment", args={"max_features":5000})
dataset.split_dataset()


"""
2) Accuracy analysis for MLP model. We will compare the workflow of MLP by using TD-IDF encoding and BOW encoding 

2.1) TD-IDF encoding 
"""
X_train_TF, X_test_TF, Y_train_TF, Y_test_TF = dataset.get_encodings(tfidf=True)

mlp_tdidf_params = mlp.find_best_mlp(X_train=X_train_TF,
                                      Y_train=Y_train_TF,
                                      X_test=X_test_TF,
                                      Y_test=Y_test_TF,
                                      n_trials = 3)


mlp_tdidf = mlp.Feedforward_Model(X_train_TF, Y_train_TF, units=mlp_tdidf_params['params_units'])
mlp_tdidf.train(LR=mlp_tdidf_params['params_learning_rate'], epochs=mlp_tdidf_params['params_epochs'])
mlp_tdidf.plot_loss_accuracy()


"""
2.1) BOW encoding 
"""

X_train_BOW, X_test_BOW, Y_train_BOW, Y_test_BOW = dataset.get_encodings()

best_mlp_bow = mlp.find_best_mlp(X_train=X_train_BOW,
                                  Y_train=Y_train_BOW,
                                  X_test=X_test_BOW,
                                  Y_test=Y_test_BOW,
                                  n_trials = 3)


























