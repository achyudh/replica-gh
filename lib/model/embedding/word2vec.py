import numpy as np
import gensim


def train_gensim(x, size=100, min_count=5):
    """
    Train a Word2Vec model from scratch with Gensim
    :param x: A list of tokenized texts (i.e. list of lists of tokens)
    :return: A trained Word2Vec model
    """
    print("Training Word2Vec...")
    model = gensim.models.Word2Vec(x, size=size, workers=8, min_count=min_count)
    model.save('data/embedding/word2vec/gensim_size%s_min%s' % (size, min_count))
    return model


def load_gensim(model_path='data/embedding/word2vec/googlenews_size300.bin', binary=True):
    if binary:
        return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    else:
        return gensim.models.Word2Vec.load(model_path)


def embedding_matrix(word_index, model_path='data/embedding/word2vec/googlenews_size300.bin', binary=True):
    if binary:
        size = int(model_path.split('.')[-2].split('/')[-1].split('_')[1][4:])
    else:
        size = int(model_path.split('/')[-1].split('_')[1][4:])
    w2v = load_gensim(model_path, binary)
    embedding_map = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        if word in w2v:
            embedding_map[i] = w2v[word]
    return embedding_map
