import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.utils import to_categorical


def read_csv(path, headers=True):
    result = list()
    with open(path) as csv_file:
        if headers:
            _temp = csv_file.readline()
        for line in csv_file:
            if line.strip() is not "":
                split_line = [x.strip() for x in line.strip().split(',')]
                if split_line[0] is not "":
                    result.append(split_line)
    return result


def to_extended_categorical(y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(n):
        if y[i] != -1:
            categorical[i, y[i]] = 1
    return categorical


def tokenize(text, stemming=True):
    english_stopwords = stopwords.words("english")
    words = [word.lower() for word in nltk.word_tokenize(text) if word.lower() not in english_stopwords]
    if stemming:
        words = [PorterStemmer().stem(word) for word in words]
    return words


def hierarchical_tokenize_and_pad(data, tokenizer=None, max_sequence_len=200, max_sequences=20,
                                  enforce_max_len=False, filter_words=False):
    temp_data = list()
    for seq in data[:,0]:
        temp_data.append(' '.join(seq.split()))
    if tokenizer is None:
        tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(temp_data)

    raw_data = list()
    max_sequences_actual = -1
    max_sequence_len_actual = -1
    for seq in data[:, 0]:
        sentences = nltk.tokenize.sent_tokenize(seq)
        raw_data.append(sentences)
        max_sequences_actual = max(len(sentences), max_sequences_actual)
        for sentence in sentences:
            word_tokens = text_to_word_sequence(sentence, filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
            max_sequence_len_actual = max(len(word_tokens), max_sequence_len_actual)

    if not enforce_max_len:
            max_sequence_len = min(max_sequence_len, max_sequence_len_actual)
            max_sequences = min(max_sequences, max_sequences_actual)

    data_x = np.zeros((len(data), max_sequences, max_sequence_len), dtype='int32')
    print("Max. Seq. Length: %d; Max Seq.: %d" %(max_sequence_len, max_sequences))

    index_filter = set()
    if filter_words:
        for word, i in tokenizer.word_index.items():
            if not (word.isalpha() or "'" in word or "-" in word):
                index_filter.add(i)

    for i, sentences in enumerate(raw_data):
        for j, sentence in enumerate(sentences):
            if j < max_sequences:
                k = 0
                word_tokens = text_to_word_sequence(' '.join(sentence.split()), filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
                for word in word_tokens:
                    if k < max_sequence_len:
                        if not filter_words or tokenizer.word_index[word] not in index_filter:
                                data_x[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
    return data_x, tokenizer, max_sequence_len, max_sequences


def tokenize_and_pad(data, tokenizer=None, max_sequence_len=400, enforce_max_len=False, filter_words=False):
    if tokenizer is None:
        tokenizer = Tokenizer(filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(data[:,0])
    raw_sequences = tokenizer.texts_to_sequences(data[:,0])

    if filter_words:
        sequences = list()
        index_filter = set()
        for word, i in tokenizer.word_index.items():
            if not (word.isalpha() or "'" in word or "-" in word):
                index_filter.add(i)
        for seq in raw_sequences:
            new_seq = list()
            for i in seq:
                if i not in index_filter:
                    new_seq.append(i)
            sequences.append(new_seq)
    else:
        sequences = raw_sequences

    seq_lengths = [len(seq) for seq in sequences]
    if not enforce_max_len:
        max_sequence_len = min(max_sequence_len, max(seq_lengths))
    data_x = pad_sequences(sequences, maxlen=max_sequence_len)
    return data_x, tokenizer, max_sequence_len
