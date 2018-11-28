import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Natural language and code embeddings for Replica-GH.")
    parser.add_argument('--word', action='store_true')
    parser.add_argument('--code', action='store_true')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline'])
    parser.add_argument('--language', type=str, default='python', choices=['python', 'java'])
    parser.add_argument('--input_path', type=str, default=os.path.join('data', 'repositories'))
    parser.add_argument('--save_path', type=str, default=os.path.join('data', 'embeddings'))
    parser.add_argument('--dim', type=int, default=300)

    args = parser.parse_args()
    return args
