import os

from unidiff import PatchSet

from lib.data import fetch
from lib.util import plot


def file_extensions(aggregate=False, save_plot=False, output_path=None):
    """

    :param aggregate:
    :param save_plot:
    :param output_path:
    :return:
    """
    # TODO: Add method for plotting results
    total_count = 0
    extension_counts = dict()
    pr_clones = fetch.pull_request_clones()
    for repo_name, orig_pr, dupl_pr in pr_clones:
        for diff in (orig_pr['diff'], dupl_pr['diff']):
            patch = PatchSet(diff)
            aggregated_extensions = set()
            for file in patch.added_files + patch.modified_files + patch.removed_files:
                extension = os.path.splitext(file.path)[1]
                if aggregate:
                    aggregated_extensions.add(extension)
                else:
                    total_count += 1
                    if extension in extension_counts:
                        extension_counts[extension] += 1
                    else:
                        extension_counts[extension] = 1

            if aggregate:
                total_count += 1
                for extension in aggregated_extensions:
                    if extension in extension_counts:
                        extension_counts[extension] += 1
                    else:
                        extension_counts[extension] = 1

    graph_labels, graph_values = list(), list()
    filter_extensions = {'.php', '.js', '.py', '.go', '.rb', '.rst', '.sh', '.java', '.cpp', '.c'}
    for extension, count in sorted(extension_counts.items(), key=lambda item: item[1], reverse=True):
        extension_counts[extension] = (count/total_count, count)
        if extension in filter_extensions:
            graph_labels.append(extension)
            graph_values.append(count/total_count)
    plot.bar_chart(graph_values, graph_labels, "File extension", "Percentage of pull requests",
                   output_path=os.path.join('data', 'plots', 'file_extension_stats.png'))

    print("Total count:", total_count)
    for extension, count_pair in sorted(extension_counts.items(), key=lambda item: item[1], reverse=True):
        if extension is None or extension == "":
            extension = "None"
        print(extension, "{0:.2f}".format(count_pair[0] * 100), count_pair[1])
