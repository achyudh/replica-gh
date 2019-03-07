import json
import os

from unidiff import PatchSet


def pull_request_clones(input_path=os.path.join('data', 'pull_requests', 'clones.json'), filter_extension=None):
    """
    Download pull request clones and build the dataset
    :param filter_extension: Filter pull request clones based on file extension
    :param input_path: path to the file containing pull request clones
    :return: dict containing the pull request clone dataset
    """
    with open(input_path, 'r') as json_file:
        pr_clones = json.load(json_file)
        if filter_extension is None:
            return pr_clones
        else:
            filtered_pr_clones = list()
            for repo_name, orig_pr, dupl_pr in pr_clones:
                diffs_with_extension = 0
                for diff in (orig_pr['diff'], dupl_pr['diff']):
                    patch = PatchSet(diff)
                    extensions = set()
                    for file in patch.added_files + patch.modified_files + patch.removed_files:
                        extensions.add(os.path.splitext(file.path)[1])
                    if filter_extension in extensions:
                        diffs_with_extension += 1
                if diffs_with_extension == 2:
                    filtered_pr_clones.append((repo_name, orig_pr, dupl_pr))
            return filtered_pr_clones


def repositories(input_path=os.path.join('data', 'repositories'), filter_extension=None):
    """

    :param input_path:
    :param filter_extension:
    :return:
    """
    code_repository = list()
    for root, dirs, files in os.walk(input_path):
        for file_path in files:
            if file_path.endswith(filter_extension):
                try:
                    with open(os.path.join(root, file_path), 'r', encoding='utf-8') as code_file:
                        code_repository.append(code_file.read())
                except UnicodeDecodeError:
                    print('UnicodeDecodeError:', os.path.join(root, file_path))
                except FileNotFoundError:
                    print('FileNotFoundError:', os.path.join(root, file_path))
    return code_repository
