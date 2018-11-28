import numpy as np
from nltk.corpus import stopwords
from scipy import sparse
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from lib.util.preprocessing import tokenize_words

english_stopwords = stopwords.words("english")


class LogisticRegression:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=english_stopwords, tokenizer=tokenize_words)
        self.classifier = linear_model.LogisticRegression(random_state=37)

    def train(self, train_x1, train_x2, train_y):
        self.vectorizer.fit(np.concatenate((train_x1, train_x2), axis=0))
        train_x1 = self.vectorizer.transform(train_x1)
        train_x2 = self.vectorizer.transform(train_x2)
        train_x = sparse.hstack((train_x1, train_x2)).A
        self.classifier.fit(train_x, train_y)

    def predict(self, predict_x1, predict_x2):
        predict_x1 = self.vectorizer.transform(predict_x1)
        predict_x2 = self.vectorizer.transform(predict_x2)
        predict_x = sparse.hstack((predict_x1, predict_x2)).A
        return self.classifier.predict(predict_x)

    def evaluate(self, evaluate_x1, evaluate_x2, evaluate_y):
        evaluate_x1 = self.vectorizer.transform(evaluate_x1)
        evaluate_x2 = self.vectorizer.transform(evaluate_x2)
        evaluate_x = sparse.hstack((evaluate_x1, evaluate_x2)).A
        predict_y = self.classifier.predict(evaluate_x)
        return {"individual": precision_recall_fscore_support(evaluate_y, predict_y),
                "micro-average": precision_recall_fscore_support(evaluate_y, predict_y, average="micro")}

    def cross_validate(self, data_x1, data_x2, data_y, num_classes, n_splits=5):
        skf = StratifiedKFold(n_splits, shuffle=False, random_state=157)
        print("Performing cross validation (%d-fold)..." % n_splits)
        precision_list = [0 for _ in range(num_classes)]
        recall_list = [0 for _ in range(num_classes)]
        f1_list = [0 for _ in range(num_classes)]
        mean_accuracy = 0
        for train_index, test_index in skf.split(data_x1, data_y):
            self.train(data_x1[train_index], data_x2[train_index], data_y[train_index])
            metrics = self.evaluate(data_x1[test_index], data_x2[test_index], data_y[test_index])
            precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
            recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
            f1_list = [x + y for x, y in zip(metrics['individual'][2], f1_list)]
            mean_accuracy += metrics['micro-average'][0]
            print("Accuracy: %s, Precision: %s, Recall: %s, F1: %s" % (metrics['micro-average'][0], metrics['individual'][0],
                                                                       metrics['individual'][1], metrics['individual'][2]))
        print("Mean accuracy: %s Mean precision: %s, Mean recall: %s, Mean F1: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list],
                                                                                      [recall/n_splits for recall in recall_list], [f1/n_splits for f1 in f1_list]))