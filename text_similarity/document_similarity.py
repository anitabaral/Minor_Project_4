import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .embedding import Embeddings


class SimilarDocuments:
    def __init__(self, data_path, model_path):

        self.documentEmbeddings = Embeddings(
            data_path, model_path
        ).document_embeddings()

    def pairwise_similarities(self):

        return cosine_similarity(self.documentEmbeddings)

    def pairwise_differences(self):

        return euclidean_distances(self.documentEmbeddings)
