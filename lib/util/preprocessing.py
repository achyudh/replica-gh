import ast
import numbers
import os

import astor
import dill as dpickle
import nltk
import numpy as np
import timeout_decorator
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from unidiff import PatchSet


def is_numeric(obj):
    """

    :param obj:
    :return:
    """
    if isinstance(obj, numbers.Number):
        return True
    elif isinstance(obj, str):
        try:
            nodes = list(ast.walk(ast.parse(obj)))[1:]
        except SyntaxError:
            return False
        if not isinstance(nodes[0], ast.Expr):
            return False
        if not isinstance(nodes[-1], ast.Num):
            return False
        nodes = nodes[1:-1]
        for i in range(len(nodes)):
            if i % 2 == 0:
                if not isinstance(nodes[i], ast.UnaryOp):
                    return False
            else:
                if not isinstance(nodes[i], (ast.USub, ast.UAdd)):
                    return False
        return True
    else:
        return False


def read_csv(path, headers=True):
    """

    :param path:
    :param headers:
    :return:
    """
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
    """

    :param y:
    :param num_classes:
    :return:
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(n):
        if y[i] != -1:
            categorical[i, y[i]] = 1
    return categorical


def get_code_changes(diff, filter_extension):
    code_changes = list()
    patch = PatchSet(diff)
    for file in patch.added_files + patch.modified_files + patch.removed_files:
        if os.path.splitext(file.path)[1] == filter_extension:
            for hunk in file:
                for line in hunk:
                    if not line.value.isspace() and not line.value.strip().startswith('#'):
                        code_changes.append(line.value)
    return ''.join(code_changes)


def tokenize_words(text, stemming=True):
    """

    :param text:
    :param stemming:
    :return:
    """
    english_stopwords = stopwords.words("english")
    words = [word.lower() for word in nltk.word_tokenize(text) if word.lower() not in english_stopwords]
    if stemming:
        words = [PorterStemmer().stem(word) for word in words]
    return words


@timeout_decorator.timeout(10)
def tokenize_code(blob, language):
    """

    :param blob:
    :param language:
    :return:
    """
    if language == 'python':
        parsed_code = astor.to_source(ast.parse(blob))
        tokenized_code = [x for x in RegexpTokenizer(r'\w+').tokenize(parsed_code) if not is_numeric(x)]
        return tokenized_code
    elif language == 'java':
        pass
    else:
        raise Exception("Unsupported language")


def tokenize_and_pad(data, tokenizer=None, max_sequence_len=500, enforce_max_len=False, filters=None,
                     filter_words=False, lower=True):
    """

    :param data:
    :param tokenizer:
    :param max_sequence_len:
    :param enforce_max_len:
    :param filter_words:
    :return:
    """
    if tokenizer is None:
        if filters is None:
            filters = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~'
        tokenizer = Tokenizer(filters=filters, lower=lower)
        tokenizer.fit_on_texts(data)
    raw_sequences = tokenizer.texts_to_sequences(data)

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


def hierarchical_tokenize_and_pad(data, tokenizer=None, max_sequence_len=200, max_sequences=20,
                                  enforce_max_len=False, filter_words=False):
    """

    :param data:
    :param tokenizer:
    :param max_sequence_len:
    :param max_sequences:
    :param enforce_max_len:
    :param filter_words:
    :return:
    """
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


def load_text_processor(input_path='title_pp.dpkl'):
    """
    Load preprocessors from disk.
    :param input_path: path of ktext.proccessor object
    :return: size of vocabulary loaded into ktext.processor, the processor
    """
    # Load files from disk
    with open(input_path, 'rb') as f:
        pp = dpickle.load(f)

    num_tokens = max(pp.id2token.keys()) + 1
    return num_tokens, pp


def load_decoder_inputs(input_path='train_title_vecs.npy'):
    """
    Load decoder inputs.
    :param input_path: The data fed to the decoder as input during training for teacher forcing.
    :return: The data that the decoder data is trained to generate (issue title).
    """
    vectorized_title = np.load(input_path)
    # For Decoder Input, you don't need the last word as that is only for prediction
    # when we are training using Teacher Forcing.
    decoder_input_data = vectorized_title[:, :-1]

    # Decoder Target Data Is Ahead By 1 Time Step From Decoder Input Data (Teacher Forcing)
    decoder_target_data = vectorized_title[:, 1:]

    return decoder_input_data, decoder_target_data


def extract_encoder_model(model):
    """
    Extract the encoder from the original Sequence to Sequence Model.
    :param model: keras model object
    :return: keras model object
    """
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def get_aux_features(pull_request):
    year = float(pull_request['created_at'][:4])
    month = float(pull_request['created_at'][5:7])
    day = float(pull_request['created_at'][8:10])
    return [pull_request['number'], pull_request['additions'], pull_request['deletions'], pull_request['commits'],
            pull_request['changed_files'], year, month, day]
