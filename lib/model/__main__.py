import os
from copy import deepcopy

import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.keras.utils import to_categorical

from lib.data import fetch
from lib.model.args import get_args
from lib.model.embedding.word import gensim
from lib.model.kim_cnn import KimCNN
from lib.model.logistic_regression import LogisticRegression
from lib.model.random_forest import RandomForest
from lib.util import preprocessing


def concatenate_str(title, body):
    return_str = ""
    if title is not None:
        return_str = title
    if body is not None:
        return_str += body
    return return_str


if __name__ == '__main__':
    # Set random seeds for Tensorflow and NumPy
    from tensorflow import set_random_seed
    from numpy.random import seed
    set_random_seed(37)
    seed(157)

    # Select GPU based on args
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model_config = deepcopy(args)

    if args.dataset == 'python_clones':
        model_config.num_classes = 2
        data_x1, data_x2, data_y = list(), list(), list()
        pr_clones = fetch.pull_request_clones(os.path.join('data', 'pull_requests', 'augmented', 'python_clones.json'))
        for orig_pr, dupl_pr, label in pr_clones:
            data_x1.append(orig_pr)
            data_x2.append(dupl_pr)
            data_y.append(label)
            data_x1, data_x2, data_y = shuffle(data_x1, data_x2, data_y, random_state=157)
        print("Dataset size:", len(data_y))
    else:
        raise Exception("Unsupported dataset")

    if args.model == 'logistic_regression':
        model = LogisticRegression()
        data_x1 = [concatenate_str(pr['title'], pr['body']) for pr in data_x1]
        data_x2 = [concatenate_str(pr['title'], pr['body']) for pr in data_x2]
        model.cross_validate(np.array(data_x1), np.array(data_x2), np.array(data_y), num_classes=args.num_classes)

    elif args.model == 'random_forest':
        model = RandomForest()
        data_x1 = [concatenate_str(pr['title'], pr['body']) for pr in data_x1]
        data_x2 = [concatenate_str(pr['title'], pr['body']) for pr in data_x2]
        model.cross_validate(np.array(data_x1), np.array(data_x2), np.array(data_y), num_classes=args.num_classes)

    elif args.model == 'kim_cnn':
        data_x1 = [concatenate_str(pr['title'], pr['body']) for pr in data_x1]
        data_x2 = [concatenate_str(pr['title'], pr['body']) for pr in data_x2]
        data_x = np.concatenate((data_x1, data_x2), axis=0)
        data_y = to_categorical(data_y)
        data_x, tokenizer, max_sequence_len = preprocessing.tokenize_and_pad(data_x)
        data_x1, _, _ = preprocessing.tokenize_and_pad(data_x1, tokenizer, max_sequence_len, enforce_max_len=True)
        data_x2, _, _ = preprocessing.tokenize_and_pad(data_x2, tokenizer, max_sequence_len, enforce_max_len=True)

        if args.word_embedding == 'google_news':
            embedding_map = gensim.embedding_matrix(tokenizer.word_index, binary=True,
                                                    model_path='data/embeddings/word/googlenews_size300.bin')
        elif args.word_embedding == 'github':
            embedding_map = gensim.embedding_matrix(tokenizer.word_index, binary=False,
                                                    model_path='data/embeddings/word/github_size300')
        else:
            raise Exception("Unsupported word embedding")

        model_config.max_sequence_len = max_sequence_len
        model = KimCNN(embedding_map, tokenizer, model_config)
        model.cross_val(data_x1, data_x2, data_y, n_splits=5)

    else:
        raise Exception("Unsupported model")
