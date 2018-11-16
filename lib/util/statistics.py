import operator
import os

from unidiff import PatchSet

from lib.data import fetch


def file_extensions():
    pr_clones = fetch.pull_request_clones()
    extension_counts = dict()
    for repo_name, orig_pr, dupl_pr in pr_clones:
        for diff in (orig_pr['diff'], dupl_pr['diff']):
            patch = PatchSet(diff)
            for file in patch.added_files + patch.modified_files + patch.removed_files:
                extension = os.path.splitext(file.path)[1]
                if extension in extension_counts:
                    extension_counts[extension] += 1
                else:
                    extension_counts[extension] = 1
    print(sorted(extension_counts, key=operator.itemgetter(1)))
