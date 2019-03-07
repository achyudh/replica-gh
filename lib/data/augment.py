import json
import os
import random

from lib.data import fetch
from lib.util.union_find import UnionFind


def pull_request_clones(output_path=os.path.join('data', 'pull_requests', 'augmented', 'clones.json'),
                        filter_extension=None):
    """
    Augment the pull request clones dataset with negative samples
    :param output_path:
    :param filter_extension:
    :return:
    """
    augmented_pr_clones = list()
    pr_clones = fetch.pull_request_clones(input_path=os.path.join('data', 'pull_requests', 'clones.json'),
                                          filter_extension=filter_extension)
    print("Dataset size before augmentation:", len(pr_clones))

    # Add PR clones to the Union Find data structure for transitive reflexive closure
    union_find = UnionFind()
    added_ids = set()
    pr_id_map = dict()
    for repo_name, orig_pr, dupl_pr in pr_clones:
        pr_id_map[orig_pr['id']] = orig_pr
        pr_id_map[dupl_pr['id']] = dupl_pr
        union_find.add(orig_pr['id'])
        union_find.add(dupl_pr['id'])
        union_find.union(orig_pr['id'], dupl_pr['id'])
        added_ids.add((orig_pr['id'], dupl_pr['id']))
        added_ids.add((dupl_pr['id'], orig_pr['id']))  # Reflexive closure
        augmented_pr_clones.append((orig_pr, dupl_pr, 1))

    while len(augmented_pr_clones) < 3 * len(pr_clones):
        id1, id2 = random.sample(pr_id_map.keys(), 2)
        if (id1, id2) not in added_ids and not union_find.connected(id1, id2):
            added_ids.add((id1, id2))
            added_ids.add((id2, id1))  # Reflexive closure
            augmented_pr_clones.append((pr_id_map[id1], pr_id_map[id2], 0))

    print("Dataset size after augmentation:", len(augmented_pr_clones))
    with open(output_path, 'w') as outfile:
        json.dump(augmented_pr_clones, outfile)
    return augmented_pr_clones
