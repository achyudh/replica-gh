from lib.util.args import get_args
from lib.util import statistics

if __name__ == '__main__':
    args = get_args()

    if args.file_extension_stats:
        # Download pull request clones and build the dataset
        statistics.file_extensions()
