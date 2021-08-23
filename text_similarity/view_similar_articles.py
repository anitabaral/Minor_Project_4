import numpy as np

from .embedding import Embeddings
from .load_data import LoadFile
from .document_similarity import SimilarDocuments


def most_similar(data_path, model_path, doc_id, matrix):
    """
    To view the top 5 similar articles.

        Parameters:
            data_path (str) = Location of the news articles
            doc_id (int) = The id of the document whose similar article needs to be printed.
            matrix (str) = The string that specifies whether to use euclidean distance or cosine similarity

        Returns:
            object: list of top 5 similar articles
    """
    similar_articles = []
    articles = LoadFile(data_path).news_dataframe()
    similar_docs = SimilarDocuments(data_path, model_path)

    if matrix == "Cosine Similarity":
        similarity_matrix = similar_docs.pairwise_similarities()
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == "Euclidean Distance":
        similarity_matrix = similar_docs.pairwise_differences()
        similar_ix = np.argsort(similarity_matrix[doc_id])
    else:
        raise ValueError('The similarity cannot be performed')
    for index in range(6):
        if index == doc_id:
            continue
        similar_articles.append(articles.iloc[index]["Initial_corpus"])

    return similar_articles
