from pathlib import Path

import gensim


class PretrainedModels:
    def __init__(self, model_path):

        self.model_path = model_path

    def gensim_model(self):

        model_w2v = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, binary=True
        )

        return model_w2v

    def glove_model(self):
        pass
