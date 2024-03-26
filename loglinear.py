import matplotlib.pyplot as plt
import numpy as np
from time import time
from tensorboardX import SummaryWriter

# Use Logistic Regression as a representative of Log-Linear Models.


class LogisticRegression:
    def __init__(
        self,
        class_num: int,
        feat_num: int,
        learning_rate: float,
        epoch_num: int
    ) -> None:
        self.class_num = class_num
        self.feat_num = feat_num
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        # The structure of the log-linear model: Wx + b = y
        self.W = np.zeros((feat_num, class_num))
        self.b = np.zeros(class_num)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        return exp_z / sum_exp_z

    def log_figure(self, values, log_num, ylabel, title, file_name):
        epochs = [i for i in range(0, self.epoch_num, log_num)]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, values, label='Training Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(file_name)
        plt.close()

    def train(self, X, y, log_num: int = -1, use_tensorboard: bool = False, use_matplotlib: bool = False):
        print("Training the model")
        start_time = time()

        N = y.shape[0]
        curr_rate = self.learning_rate
        thresh1 = int(self.epoch_num * 0.8)
        thresh2 = int(self.epoch_num * 0.95)
        if use_tensorboard:
            writer = SummaryWriter(log_dir='logs')
        loss_record = []
        accuracy_record = []
        for i in range(self.epoch_num):
            # (N, F) * (F, C) -> (N, C)
            y_pred = self.softmax(np.dot(X, self.W) + self.b)
            log_probs = np.log(y_pred)  # (N, C)
            y_true = np.zeros((N, self.class_num))
            y_true[np.arange(N), y - 1] = 1  # (N, C)
            loss = -np.sum(log_probs * y_true) / N  # int

            # The update algorithm with changing learning rate
            # (F, N) * (N, C) -> (F, C)
            dW = np.dot(X.T, (y_pred - y_true)) / N
            db = np.sum(y_pred - y_true, axis=0) / N  # C
            self.W -= curr_rate * dW
            self.b -= curr_rate * db
            if log_num > 0:
                if i % log_num == 0:
                    accuracy = np.sum((np.argmax(y_pred, axis=1) + 1) == y) / N
                    print("Iteration {}: loss={}\t train accuracy={}".format(
                        i, loss, accuracy))
                    loss_record.append(loss)
                    accuracy_record.append(accuracy)
                    if use_tensorboard:
                        writer.add_scalar('Training Loss lr={} thresh1={} thresh2={}'.format(
                            self.learning_rate, thresh1, thresh2), loss, i)
                        writer.add_scalar('Training Accuracy lr={} thresh1={} thresh2={}'.format(
                            self.learning_rate, thresh1, thresh2), accuracy, i)
            if i == thresh1:
                curr_rate /= 5
            if i == thresh2:
                curr_rate /= 2
        print("Time consumption: {:.2f} sec".format(time()-start_time))
        if use_tensorboard:
            writer.close()
        if use_matplotlib:
            self.log_figure(loss_record, log_num=log_num, ylabel='Loss', title='Training Loss',
                            file_name='Training Loss lr={} thresh1={} thresh2={}.png'.format(self.learning_rate, thresh1, thresh2))
            self.log_figure(accuracy_record, log_num=log_num, ylabel='Accuracy', title='Training Accuracy',
                            file_name='Training Accuracy lr={} thresh1={} thresh2={}.png'.format(self.learning_rate, thresh1, thresh2))

    def predict(self, X):
        y_pred = self.softmax(np.dot(X, self.W) + self.b)
        return np.argmax(y_pred, axis=1) + 1


# logistic_regression = LogisticRegression(
#     class_num=4, feat_num=5, learning_rate=0.1, epoch_num=1000, log_num=100)
# X = np.array([[1, 2, 3, 4, 5],
#               [6, 7, 8, 9, 10],
#               [0, 1, 2, 3, 4]])
# y = np.array([1, 2, 3])
# N = 3
# feat_num = 5
# class_num = 4
# logistic_regression.train(X, y)
# _X = np.array([[1, 2, 3, 4, 5],
#               [6, 7, 8, 9, 10],
#               [0, 1, 2, 3, 4]])
# print(logistic_regression.predict(_X))
