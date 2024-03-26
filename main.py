import numpy as np
from preprocess import Preprocessor
from tfidf import FeatureExtractor
from loglinear import LogisticRegression
from evaluation import Evalutaion

if __name__ == "__main__":
    # Load the train data
    with open("./ag_news_csv/classes.txt", "r") as class_file:
        content = class_file.read()
        classes = [x for x in content.split('\n') if x]
    class_num = len(classes)

    # Data preprocessing
    words_train, label_train = Preprocessor()("./ag_news_csv/train.csv",
                                              scale=0.05, concat_title=True)
    words_test, label_test = Preprocessor()("./ag_news_csv/test.csv",
                                            scale=0.05, concat_title=True)

    # Feature extraction
    feature_num = 5000
    feature_extractor = FeatureExtractor()
    f_train = feature_extractor.TFIDF(
        words_train, num=feature_num, is_train=True)
    f_test = feature_extractor.TFIDF(
        words_test, num=feature_num, is_train=False)

    # The implementation of log-linear model with the update algorithm
    y_train = np.array(label_train)
    y_test = np.array(label_test)
    logistic_regression = LogisticRegression(
        class_num=class_num, feat_num=feature_num, learning_rate=0.3, epoch_num=10000)
    logistic_regression.train(
        f_train, y_train, log_num=100, use_tensorboard=True, use_matplotlib=True)

    # Evaluation
    test_size = y_test.shape[0]
    sample = np.random.choice(np.arange(y_train.shape[0]), size=test_size)
    f_train_sample = f_train[sample]
    y_train_sample = y_train[sample]
    y_train_pred = logistic_regression.predict(f_train_sample)
    Evalutaion().accuracy(y_true=y_train_sample, y_pred=y_train_pred, name="Train")
    Evalutaion().macro_f1_score(y_true=y_train_sample,
                                y_pred=y_train_pred, class_num=class_num, name="Train")

    y_test_pred = logistic_regression.predict(f_test)
    Evalutaion().accuracy(y_true=y_test, y_pred=y_test_pred, name="Test")
    Evalutaion().macro_f1_score(y_true=y_test, y_pred=y_test_pred,
                                class_num=class_num, name="Test")
