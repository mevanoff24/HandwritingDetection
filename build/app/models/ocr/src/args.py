import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog='src')
    parser.add_argument('--train', '-t', action='store_true',
                        help='train or not')
    parser.add_argument('--use_s3', '-s', default=False, type=str,
                        help='specify input file')
    parser.add_argument('--config-file', '-c', default='config.py', type=str,
                        help='specify config file')
    parser.add_argument('--save_path', '-w', default='models/embedding.vec',
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m', default='models/model.param',
                        help='model output filename')
    parser.add_argument('--task', default='', type=str,
                        help='choose evaluation task from [mscc]')

    return parser.parse_args()