import os

import numpy as np

from lib.data import fetch
from lib.model.args import get_args
from lib.model.logistic_regression import LogisticRegression
from sklearn.utils import shuffle


def concatenate_str(title, body):
    return_str = ""
    if title is not None:
        return_str = title
    if body is not None:
        return_str += body
    return return_str


if __name__ == '__main__':
    args = get_args()
    if args.dataset == 'python_clones':
        num_classes = 2
        data_x1, data_x2, data_y = list(), list(), list()
        pr_clones = fetch.pull_request_clones(os.path.join('data', 'pull_requests', 'augmented', 'python_clones.json'))
        for orig_pr, dupl_pr, label in pr_clones:
            data_x1.append(orig_pr)
            data_x2.append(dupl_pr)
            data_y.append(label)
            data_x1, data_x2, data_y = shuffle(data_x1, data_x2, data_y, random_state=157)
    else:
        raise Exception("Unsupported dataset")

    if args.model == 'logistic_regression':
        model = LogisticRegression()
        data_x1 = [concatenate_str(pr['title'], pr['body']) for pr in data_x1]
        data_x2 = [concatenate_str(pr['title'], pr['body']) for pr in data_x2]
        model.cross_validate(np.array(data_x1), np.array(data_x2), np.array(data_y), num_classes=num_classes)
    else:
        raise Exception("Unsupported model")
