import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Text_Dataset:
    def __init__(self, df, col_text=None, col_label=None, args={}):
        self.df = df.copy() # we shall work on pandas df

        col_text = col_text or self.df.columns[0]
        col_label = col_label or self.df.columns[1]

        if col_text not in self.df.columns or col_label not in self.df.columns:
            raise ValueError(f"Column '{col_text}' or '{col_label}' not found in DataFrame columns: {list(self.df.columns)}")

        self.documents = self.df[col_text]
        self.labels = self.df[col_label]

        if not isinstance(self.documents, pd.Series):
            raise ValueError("Expected text column to be a Series of strings, got:", type(self.documents))

        self.bow = CountVectorizer(**args).fit(self.documents)
        self.tfidf = TfidfVectorizer(**args).fit(self.documents)

        self.label_enc = LabelEncoder().fit(self.labels)
        self.label_enc = LabelEncoder().fit(self.labels)


    def split_dataset(self, test_size=0.2):
        X_train, X_test, Y_train, Y_test = train_test_split(self.documents, self.labels, test_size=test_size, stratify=self.labels)
        self.split = {'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test}


    def get_encodings(self, tdidf=False):
        """
        Data for MLP models.
        :param tdidf: encoding type
        :return: Train and test sets as BOW or TFIDF vectors.
        """
        encoder = self.bow # BOW (default)
        if tdidf: # TF-IDF
            encoder = self.tfidf

        # transform (encode) documents
        X_train = encoder.transform(self.split['X_train']).toarray()
        X_test = encoder.transform(self.split['X_test']).toarray()
        Y_train = self.label_enc.transform(self.split['Y_train'])
        Y_test = self.label_enc.transform(self.split['Y_test'])
        return X_train, X_test, Y_train, Y_test


    def get_sequences(self, vocab_size=1000, maxlen=100):
        """
        Introduce the tokenizer class to easily produce sequences for RRN models.
        :param vocab_size: the size of the vocabulary.
        :param maxlen: the maximum length of the sequences.
        :return: Train and test sets.
        """
        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=maxlen
        )

        # fit tokenizer on the training set
        self.tokenizer.adapt(self.split['X_train'])

        # vocabulary
        self.vocabulary = self.tokenizer.get_vocabulary()

        # return indexes
        X_train = self.tokenizer(self.split['X_train']).numpy()
        X_test = self.tokenizer(self.split['X_test']).numpy()

        # encode for binary classification
        Y_train = self.label_enc.transform(self.split['Y_train']).reshape(-1, 1)
        Y_test = self.label_enc.transform(self.split['Y_test']).reshape(-1, 1)

        return X_train, X_test, Y_train, Y_test
        

    def term_frequency_table(self):
        # take all texts and transform it to array
        data = self.bow.transform(self.documents)

        # count occurances
        sum_of_occurences = data.toarray().sum(axis=0)

        # create dict with occurences
        terms = self.bow.get_feature_names_out()
        terms_frequency = {}
        for i in range(len(terms)):
            terms_frequency[terms[i]] = sum_of_occurences[i]

        # create df
        df_terms = pd.DataFrame(list(terms_frequency.items()), columns=['Term', 'Frequency'])

        # group
        df_terms = df_terms.groupby('Frequency').agg({'Term': list})
        df_terms['Number of terms'] = [len(term_list) for term_list in df_terms['Term']]
        df_terms = df_terms.reset_index()
        df_terms = df_terms[['Frequency', 'Number of terms', 'Term']]
        return df_terms
