import pandas as pd


def tag_with_spacy(doc):
    clean_tokens = []
    original_tokens = []
    for token in doc:
        if not (token.is_punct or
                not token.is_alpha or
                token.is_space or
                token.like_url):
            clean_tokens.append(token.lemma_)
        original_tokens.append(token.text)
    original_text = ' '.join(original_tokens)
    return clean_tokens, original_text


def prepare_df(df, nlp):
    texts = df['text'].tolist()
    sentiments = df['sentiment'].tolist()
    rows = []
    for text, sentiment in zip(texts, sentiments):
        doc = nlp(text)
        clean_text, original_text = tag_with_spacy(doc)
        rows.append({
            'sentiment': sentiment,
            "clean_text": clean_text,
            "clean_text_str": ' '.join(clean_text),
            "text": original_text})
    df = pd.DataFrame(rows)
    return df