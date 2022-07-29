import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    parser.add_argument('--mode', default = 'train', choices = ['train', 'test'], help = '')

    # For Loader
    parser.add_argument('--sources_path', default = 'datasources', help = '')
    parser.add_argument('--targets_path', default = 'datatargets', help = '')

    parser.add_argument('--train_file', default = 'train.tsv', help = '')
    parser.add_argument('--valid_file', default = 'valid.tsv', help = '')
    parser.add_argument('--test_file' , default = 'test.tsv' , help = '')

    parser.add_argument('--min_freq', type = int, default = 5, help = '')
    parser.add_argument('--max_numb', type = int, default = 10000, help = '')
    parser.add_argument('--max_seq_len', type = int, default = 32, help = '')

    # For Module
    parser.add_argument('--emb_dim', type = int, default = 512, help = '')
    parser.add_argument('--hid_dim', type = int, default = 512, help = '')

    parser.add_argument('--dropout', type = float, default = 0.2, help = '') # !

    # For Train
    parser.add_argument('--batch_size', type = int, default = 64, help = '')
    parser.add_argument('--num_epochs', type = int, default = 10, help = '')

    parser.add_argument('--grad_clip', type = float, default = 0.1, help = '') # !

    parser.add_argument('--lr', type = float, default = 0.0001, help = '') # !

    # For Test
    parser.add_argument('--decoding_algorithm', default = 'temperature_sampling', choices = ['temperature_sampling', 'top_k_sampling', 'top_p_sampling', 'beam_search'], help = '')

    parser.add_argument('--T', type = float, default = 1e-13, help = '')
    parser.add_argument('--K', type = int, default = 10, help = '')
    parser.add_argument('--P', type = float, default = 0.3, help = '')
    parser.add_argument('--B', type = int, default = 5, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
