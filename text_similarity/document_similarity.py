import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .embedding import Embeddings
from .load_data import LoadFile

class DocumentSimilarity:

    def __init__(self):
        self.documentEmbeddings = Embeddings.document_embeddings()
    
    def cosine_similarities(self):

        return cosine_similarity(self.documentEmbeddings)

    def euclidean_distances(self, document_embeddings):

        return euclidean_distances(self.documentEmbeddings)

