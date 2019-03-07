import os
import json

from lib.data import github, fetch


def pull_request_clones(input_path=os.path.join('data', 'pull_requests', 'clone_ids.csv'),
                        output_path=os.path.join('data', 'pull_requests', 'clones.json')):
    """
    Download pull request clones and build the dataset
    :param input_path: path to the file containing pull request clone IDs
    :param output_path: path to save the pull request clone dataset
    """
    pr_clones = fetch.pull_request_clones()
    downloaded_prs = set()
    for repo_name, orig_pr, dupl_pr in pr_clones:
        if (repo_name, orig_pr['number'], dupl_pr['number']) in downloaded_prs:
            raise Exception("Duplicate entries in the dataset")
        downloaded_prs.add((repo_name, orig_pr['number'], dupl_pr['number']))

    missing_prs = set()
    with open(input_path, 'r') as csv_file:
        ctr = 0
        for line in csv_file:
            repo_owner, repo_name, orig_pr_id, dupl_pr_id = line.split(',')
            orig_pr_id, dupl_pr_id = int(orig_pr_id), int(dupl_pr_id)
            repo_name = repo_owner + '/' + repo_name
            if (repo_name, orig_pr_id, dupl_pr_id) not in downloaded_prs:
                try:
                    orig_pr = github.pull_request(repo_name, orig_pr_id, get_diff=True)
                    dupl_pr = github.pull_request(repo_name, dupl_pr_id, get_diff=True)
                except Exception as e:
                    print(e)
                    missing_prs.add((repo_name, orig_pr_id, dupl_pr_id))
                    continue
                pr_clones.append((repo_name, orig_pr, dupl_pr))
                ctr += 1
                if ctr == 500:
                    break
    print("Number of pull requests downloaded:", len(pr_clones))
    if len(missing_prs) > 0:
        print("Missing pull requests:", missing_prs)
    with open(output_path, 'w') as outfile:
        json.dump(pr_clones, outfile)