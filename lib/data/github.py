from requests.auth import HTTPBasicAuth
import requests, json, time
from flask import Flask

with open('config.json', 'r') as config_file:
    client_config = json.load(config_file)

app = Flask(__name__)
http_auth_username = client_config['HTTP_AUTH_USERNAME']
http_auth_secret = client_config['HTTP_AUTH_SECRET']
http_auth = HTTPBasicAuth(http_auth_username, http_auth_secret)

# Header to get reactions along with comments
reactions_header = {'Accept': 'application/vnd.github.squirrel-girl-preview',
                    'direction': 'desc', 'sort': 'created'}
# Header to get diff along with pull request
diff_header = {'Accept': 'application/vnd.github.VERSION.diff'}


def repo(repo_id):
    """
    This uses an undocumented GitHub API endpoint. Could be deprecated in the near future without notice.
    :param repo_id: integer identifier for the repository
    :return: dict containing repository metadata
    """
    try:
        request_url = 'https://api.github.com/repositories/%s' % repo_id
    except Exception as e:
        raise e
    return generic(request_url)


def rate_reset_wait(headers):
    """

    :param headers:
    :return:
    """
    if 'X-RateLimit-Remaining' in headers:
        ratelimit_remaining = int(headers['X-RateLimit-Remaining'])
    else:
        ratelimit_remaining = 1
    if ratelimit_remaining <= 0:
        print("Waiting for %d minutes..." % ((int(headers['X-RateLimit-Reset']) - time.time())//60))
        time.sleep(int(headers['X-RateLimit-Reset']) - time.time() + 1)
        return "RateLimit Reset"
    else:
        if ratelimit_remaining % 100 == 0:
            print('X-RateLimit-Remaining:', ratelimit_remaining)
        return "Positive RateLimit"


def generic(request_url, headers=None, plaintext=False):
    """

    :param request_url:
    :param headers:
    :return:
    """
    if headers is not None:
        response = requests.get(request_url, auth=http_auth, headers=headers)
    else:
        response = requests.get(request_url, auth=http_auth)
    wait_status = rate_reset_wait(response.headers)
    if wait_status != "Positive RateLimit":
        if headers is not None:
            response = requests.get(request_url, auth=http_auth, headers=headers)
        else:
            response = requests.get(request_url, auth=http_auth)
    if response.status_code == 404:
        raise Exception(response.json()['message'])
    if plaintext:
        return response.content.decode("utf-8", "ignore")
    else:
        return response.json()


def pull_request(repo_name, pr_number, get_diff=False):
    """

    :param repo_name: string in owner/repo_name format
    :param pr_number: integer identifier for the pull request
    :param get_diff: gets the diff associated with the pull request if true
    :return: dict containing the pull request meta data along with the diff if get_diff was true
    """
    request_url = 'https://api.github.com/repos/%s/pulls/%s' % (repo_name, pr_number)
    try:
        response = generic(request_url)
        if get_diff:
            response['diff'] = generic(request_url, diff_header, plaintext=True)
    except Exception as e:
        raise e
    return response

