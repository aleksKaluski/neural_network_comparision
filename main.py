import pandas as pd
import numpy as np
import spacy
import os
from pathlib import Path

from source import prepare_data
"""
Preprocessing
"""
# load models and dataset
df = pd.read_csv("hf://datasets/gxb912/large-twitter-tweets-sentiment/train.csv", nrows=100)
nlp = spacy.load("en_core_web_sm")

# prepare dataset
df = prepare_data.prepare_df(df, nlp)
dataset = Text_Dataset(df, col_text="clean_text_str", col_label="sentiment", args={"max_features":5000})
dataset.split_dataset()



