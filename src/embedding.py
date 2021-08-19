import numpy as numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

from .Load_data import LoadFile

class Embeddings:
    def __init__(self):
        self.news_df = LoadFile.news_dataframe


    def get_tokenized_elements():

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(news_dataframe.Cleaned_corpus)
        tokenized_documents = tokenizer.texts_to_sequences(news_dataframe.Cleaned_corpus)
        tokenized_paded_documents = pad_sequences(tokenized_documents, maxlen=64, padding='post')

        return tokenizer, tokenized_paded_documents

    def get_tfidf_elements():

        tfidfvectoriser = TfidfVectorizer(max_features = 3000)
        tfidfvectoriser.fit(news_dataframe.Cleaned_corpus)
        tfidf_vectors = tfidfvectoriser.transform(news_dataframe.Cleaned_corpus)
        tfidf_vectors = tfidf_vectors.toarray()
        words = tfidfvectoriser.get_feature_names()

        return words, tfidf_vectors

    def document_word_embeddings():

        tokenizer, tokenized_paded_documents = get_tokenized_elements()
        vocab_size = len(tokenizer.word_index)+1
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, i in tokenizer.word_index.items():
            if word in model_w2v:
            embedding_matrix[i] = model_w2v[word]  
        doc_word_embeddings = np.zeros((len(tokenized_paded_documents), 3000, 300))
        for i in range(len(tokenized_paded_documents)):
            for j in range(len(tokenized_paded_documents[0])):
                doc_word_embeddings[i][j] = embedding_matrix[tokenized_paded_documents[i][j]]

        return embedding_matrix, doc_word_embeddings

    def document_embeddings():

        tokenizer, tokenized_paded_documents = get_tokenized_elements()
        words, tfidf_vectors = get_tfidf_elements()
        embedding_matrix, doc_word_embeddings = document_word_embeddings()
        documentEmbeddings = np.zeros((len(tokenized_paded_documents), 300))
        for i in range(len(doc_word_embeddings)):
            for j in range(len(words)):   
            documentEmbeddings[i] += embedding_matrix[tokenizer.word_index[words[j]]] * tfidf_vectors[i][j]       
        documentEmbeddings = documentEmbeddings / np.sum(tfidf_vectors, axis = 1).reshape(-1, 1)
  
        return documentEmbeddings
