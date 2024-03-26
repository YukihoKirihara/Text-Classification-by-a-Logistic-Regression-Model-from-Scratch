import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class Evalutaion:
    def __init__(self) -> None:
        pass

    def accuracy(self, y_true, y_pred, name: str):
        N = y_true.shape[0]
        print("Evaluation of {}, size={}, function=accuracy".format(name, N))
        my_accuracy = np.sum(y_true == y_pred) / N
        sklearn_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        print("Accuracy={:.4f}\t sklearn.metrics.accuracy_score={:.4f}".format(
            my_accuracy, sklearn_accuracy))

    def macro_f1_score(self, y_true, y_pred, class_num: int, name: str):
        N = y_true.shape[0]
        print("Evaluation of {}, size={}, function=macro_f1_score".format(name, N))
        precisions = []
        recalls = []

        for c in range(1, class_num+1):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        f1_scores = [2 * (p * r) / (p + r) if (p + r) >
                     0 else 0 for p, r in zip(precisions, recalls)]
        macro_f1score = np.mean(f1_scores)
        sklearn_f1score = f1_score(
            y_true=y_true, y_pred=y_pred, average='macro')
        print("F1_Score={:.4f}\t sklearn.metrics.f1_score={:.4f}".format(
            macro_f1score, sklearn_f1score))
