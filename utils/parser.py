import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Module
    parser.add_argument('--emb_dim', type = int, default = 512, help = '')
    parser.add_argument('--hid_dim', type = int, default = 512, help = '')

    parser.add_argument('--dropout', type = float, default = 0.5, help = '')

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    parser.add_argument('--grad_clip', type = float, default = 1, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
