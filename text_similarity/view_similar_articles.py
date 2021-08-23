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
            None
    """
    articles = LoadFile(data_path).news_dataframe()
    similar_docs = SimilarDocuments(data_path, model_path)
    print(f'Document: {articles.iloc[doc_id]["Initial_corpus"]}')
    print("\n")
    print("Similar Documents:")
    if matrix == "Cosine Similarity":
        similarity_matrix = similar_docs.pairwise_similarities()
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix == "Euclidean Distance":
        similarity_matrix = similar_docs.pairwise_differences()
        similar_ix = np.argsort(similarity_matrix[doc_id])
    for index in range(5):
        if index == doc_id:
            continue
        print("\n")
        print(f'Document: {articles.iloc[index]["Initial_corpus"]}')
        print(f"{matrix} : {similarity_matrix[doc_id][index]}")

    return None
