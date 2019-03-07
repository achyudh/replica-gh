import os

from lib.data.args import get_args
from lib.data import download, augment

if __name__ == '__main__':
    args = get_args()

    if args.download_clones:
        # Download pull request clones and build the dataset
        download.pull_request_clones()
    if args.augment_clones:
        augment.pull_request_clones(output_path=os.path.join('data', 'pull_requests', 'augmented', 'clones.json'),
                                    filter_extension=args.filter_extension)
