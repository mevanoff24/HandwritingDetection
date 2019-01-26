import dill as pickle


def unpickle(filename):
    """Unpickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

