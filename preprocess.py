import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pandas
import re
from time import time
from typing import List, Tuple


class Preprocessor:
    # TODO: Should time-related terms like 'today', 'Monday' be added?
    MY_STOP_WORDS = ['u', 'said', 'reuter', 'ap', 'wa',
                     'gt', 'lt', 'quot', 'would', 'afp', 'could', 'may']

    def __init__(self) -> None:
        pass

    def __call__(self, url, scale=1.0, concat_title=False) -> Tuple[List[List[str]], List[int]]:
        print("Preprocessing", url.split('/')[-1])
        start_time = time()

        df = pandas.read_csv(url, header=None, names=[
                             "Label", "Title", "Content"])

        # Choose a subset as train set
        if scale < 1.0:
            size = int(df.size * scale)
            df = df.sample(size)

        # Concatenate the title and the content
        if concat_title:
            df["Title"].apply(lambda x: x + " ")
            df["Content"] = df["Title"] + df["Content"]
        df.drop(columns=['Title'], axis=1, inplace=True)

        nltk.data.path = [os.getcwd() + '/nltk_data']  # Use pre-download data
        _stopwords = stopwords.words('english')
        _stopwords += self.MY_STOP_WORDS
        porter_stemmer = PorterStemmer()
        lemmarizer = WordNetLemmatizer()

        def content_filter(t):
            if not t:
                return ''
            # Remove punctuations and numbers
            t = t.translate(
                {ord(c): " " for c in "!@#$%^&*()[]''{};:,./<>?\|`~-=_+"})
            t = re.sub(r"[\"\`0-9]", '', t)

            t = t.lower()
            # Remove URLs
            t = re.sub(r'http\S+', '', t)

            # Split into words
            words = word_tokenize(t)

            # Stem and Lemmarize
            lemma_words = []
            for w in words:
                _w = lemmarizer.lemmatize(porter_stemmer.stem(w))
                if _w not in _stopwords:
                    lemma_words.append(_w)

            return lemma_words

        df['Content'] = df['Content'].apply(content_filter)
        print("Time consumption: {:.2f} sec".format(time()-start_time))
        print(df.head)

        X = df['Content'].tolist()
        Y = df['Label'].tolist()

        return X, Y
