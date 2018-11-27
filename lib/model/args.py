from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Machine learning models for Replica-GH.")
    parser.add_argument('--model', type=str, default='static', choices=['logistic_regression', 'random_forest', 'kim_cnn'])
    parser.add_argument('--gpu', type=int, default=0)  # If the value is -1, use CPU
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='python_clones', choices=['python_clones'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='data/checkpoints')
    parser.add_argument('--word-embedding', type=str, default='google_news', choices=['google_news', 'github'])
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--code-embedding', type=str, default='baseline', choices=['baseline'])
    parser.add_argument('--code-embedding-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    return args
