from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Machine learning models for Replica-GH.")
    parser.add_argument('--model', type=str, default='static', choices=['logistic_regression'])
    parser.add_argument('--gpu', type=int, default=0)  # If the value is -1, use CPU
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--bottleneck_layer', action='store_true')
    parser.add_argument('--single_label', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='python_clones', choices=['python_clones'])
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='data/checkpoints')
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    return args
