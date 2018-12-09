import gensim
from timeout_decorator.timeout_decorator import TimeoutError

from lib.data import fetch
from lib.util import preprocessing


def train(input_path, language, dim, min_count):
    """

    :param input_path:
    :param language:
    :param dim:
    :param min_count:
    :return:
    """
    language_extension_map = {'python': '.py', 'java': '.java'}
    code_repository = fetch.repositories(input_path, filter_extension=language_extension_map[language])
    tokenized_repository = list()

    for blob in code_repository:
        try:
            tokenized_repository.append(preprocessing.tokenize_code(blob, language))
        except TimeoutError:
            print('TimeoutError:', ' '.join(blob[:50].split()) + "...")
        except Exception as e:
            print(type(e).__name__ + ":", ' '.join(blob[:50].split()) + "...")
        if len(tokenized_repository) % 1000 == 0:
            print("Tokenized %d of %d blobs..." % (len(tokenized_repository), len(code_repository)))

    print('Size of code repository:', len(code_repository))
    print("Training baseline code embeddings...")
    model = gensim.models.Word2Vec(tokenized_repository, size=dim, workers=8, min_count=min_count)
    model.save('data/embeddings/code/baseline_size%s_min%s' % (dim, min_count))
    return model


def load():
    pass


def embedding_matrix():
    pass
