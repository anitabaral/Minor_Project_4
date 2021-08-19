from pathlib import Path

import gensim

class LoadModel:
    def __init__(self):
        pass
    
    def gensim_model(self):
        
        data_folder = Path("/content/drive/MyDrive/Leapfrog_internship/Project 6/")
        file_path = data_folder / "GoogleNews-vectors-negative300.bin.gz"
        model_w2v = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

        return model_w2v