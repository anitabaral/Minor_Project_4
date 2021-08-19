import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from .Embeddings import Embeddings
from .Load_data import LoadFile

class Document_similarity:
    def __init__(self):
        pass
    
    def cosine_similarities(self, document_embeddings):

        return cosine_similarity(document_embeddings)

    def euclidean_distances(self, document_embeddings):

        return euclidean_distances(document_embeddings)
    
    def most_similar(news_dataframe, oc_id, top_similar, similarity_matrix, matrix):
            print (f'Document: {news_dataframe.iloc[doc_id]["Initial_corpus"]}')
            print ('\n')
            print ('Similar Documents:')
            if matrix == 'Cosine Similarity':
                similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
            elif matrix == 'Euclidean Distance':
                similar_ix = np.argsort(similarity_matrix[doc_id])
            for ix in range(top_similar):
                if ix == doc_id:
                    continue
                print('\n')
                print (f'Document: {news_dataframe.iloc[ix]["Initial_corpus"]}')
                print (f'{matrix} : {similarity_matrix[doc_id][ix]}')