import argparse


def parse_args():
    """Command line arguments from Context2Vec model"""
    parser = argparse.ArgumentParser(prog='src')
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--train-file', '-t', type=str, help='specify input file')
    parser.add_argument('--val-file', '-v', type=str, help='specify input file')
    parser.add_argument('--config-file', '-c', default='./config.toml', type=str, help='specify config toml file')
    parser.add_argument('--wordsfile', '-w', default='models/embedding.vec', type=str, help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m', default='models/model.param', type=str, help='model output filename')
    parser.add_argument('--log-filename', '-l', default='logs.txt', type=str, help='model log filename')
    parser.add_argument('--use-s3', '-s', default='false', type=str, help='use data stored in S3')
    parser.add_argument('--load', '-r', default='false', type=str, help='load saved model')
    parser.add_argument('--patience', '-p', default=5, type=int, help='early stopping threshold')
    return parser.parse_args()