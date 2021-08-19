import sys
sys.path.append('..')
from pathlib import Path

import gensim

class LoadModel:
    def __init__(self):
        self.this_dir, self.this_file = os.path.split(__file__)
    
    def gensim_model(self):
        
        data_folder = Path(self.this_dir)
        file_path = data_folder / "model" / "GoogleNews-vectors-negative300.bin.gz"
        model_w2v = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

        return model_w2v