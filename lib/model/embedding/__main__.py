import os

from lib.model.embedding import word, code
from lib.model.embedding.args import get_args


if __name__ == '__main__':
    args = get_args()
    if args.code:
        if args.model == 'baseline':
            code.baseline.train(os.path.join(args.input_path, args.language), args.language, args.dim, args.min_count)
        else:
            raise Exception("Unsupported model")
    elif args.word:
        if args.model == 'baseline':
            # TODO: Add training data as param to gensim.train()
            word.gensim.train()
        else:
            raise Exception("Unsupported model")
    else:
        raise Exception("Unsupported mode")
