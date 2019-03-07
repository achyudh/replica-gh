from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Data processing module for Replica-GH.")
    parser.add_argument('--download-clones', action='store_true',
                        help='download pull request clones and build the dataset')
    parser.add_argument('--augment-clones', action='store_true',
                        help='augment the pull request clones dataset with negative samples')
    parser.add_argument('--filter-extension', type=str, default=None,
                        help='filter pull request diffs by extension')
    args = parser.parse_args()
    return args
