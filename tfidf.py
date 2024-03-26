import numpy as np
from numpy.typing import NDArray
from time import time
from typing import List


class FeatureExtractor:
    def __init__(self) -> None:
        # Get the feature words in train set and store for the test set.
        self.feat_words = []
        pass

    def TFIDF(self, words: List[List[str]], num: int = 5000, is_train: bool = True) -> NDArray:
        N = len(words)
        print("Extracting features: TFIDF, number of texts={}".format(N))
        start_time = time()

        feat_words = []
        if is_train:
            print("Selecting the feature words")
            # Limit the number of features to the `num` most frequent
            word_count = dict()  # The number of texts that contains word t
            for ws in words:
                for w in ws:
                    if w in word_count.keys():
                        word_count[w] += 1
                    else:
                        word_count[w] = 1
            feat_words = sorted(
                word_count, key=word_count.get, reverse=True)[:num]
            self.feat_words = feat_words
            print("Selected {} feature words:".format(
                len(feat_words)), feat_words[:10], "...")
        else:
            feat_words = self.feat_words

        M = len(feat_words)
        feat_words_set = set(feat_words)
        TF = np.zeros((N, M))  # TF(d, t)
        IDF_bottom = np.zeros(M)
        for i, ws in enumerate(words):
            bottom = len(ws)
            TF_dict = dict()
            for w in ws:
                if w in feat_words_set:
                    if w in TF_dict.keys():
                        TF_dict[w] += 1
                    else:
                        TF_dict[w] = 1
            for j, w in enumerate(feat_words):
                cnt = TF_dict.get(w, 0)
                TF[i][j] = cnt / bottom
                if cnt > 0:
                    IDF_bottom[j] += 1
        IDF = np.log((N + 1) / (IDF_bottom + 1)) + 1
        TFIDF = TF * IDF
        l2_norm = np.linalg.norm(TFIDF, axis=1, keepdims=True)
        TFIDF = TFIDF / l2_norm

        print("Time consumption: {:.2f} sec".format(time()-start_time))
        print("The shape of TFIDF:", TFIDF.shape)

        return TFIDF
