import toml


class Config:
    def __init__(self, filename):
        self.filename = filename
        config = toml.load(self.filename)
        
        model = config.get('model', {})
        self.word_embed_size = int(model.get('word_embed_size', 300))
        self.hidden_size = int(model.get('hidden_size', 300))
        self.n_layers = int(model.get('n_layers', 1))
        self.dropout = float(model.get('dropout', 0.0))

        train = config.get('train', {})
        self.n_epochs = int(train.get('n_epochs', 1))
        self.batch_size = int(train.get('batch_size', 100))
        self.min_freq = int(train.get('min_freq', 1))
        self.ns_power = float(train.get('ns_power', 0.75))
        self.learning_rate = float(train.get('learning_rate', 1e-4))


        
        
        
if __name__ == '__main__':
    config = Config('./config.toml')
    print(config)