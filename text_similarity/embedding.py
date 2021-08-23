import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

from .load_data import LoadFile
from .load_model import PretrainedModels


class Embeddings:
    def __init__(self, data_path, model_path):

        self.news_df = LoadFile(data_path).news_dataframe()
        self.model_w2v = PretrainedModels(model_path).gensim_model()

    def get_tokenized_elements(self):

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.news_df.Cleaned_corpus)
        tokenized_documents = tokenizer.texts_to_sequences(self.news_df.Cleaned_corpus)
        tokenized_paded_documents = pad_sequences(
            tokenized_documents, maxlen=64, padding="post"
        )

        return tokenizer, tokenized_paded_documents

    def get_tfidf_elements(self):

        tfidfvectoriser = TfidfVectorizer(max_features=3000)
        tfidfvectoriser.fit(self.news_df.Cleaned_corpus)
        tfidf_vectors = tfidfvectoriser.transform(self.news_df.Cleaned_corpus)
        tfidf_vectors = tfidf_vectors.toarray()
        words = tfidfvectoriser.get_feature_names()

        return words, tfidf_vectors

    def document_word_embeddings(self):

        tokenizer, tokenized_paded_documents = self.get_tokenized_elements()
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, index in tokenizer.word_index.items():
            if word in self.model_w2v:
                embedding_matrix[index] = self.model_w2v[word]
        doc_word_embeddings = np.zeros((len(tokenized_paded_documents), 3000, 300))
        for doc_index in range(len(tokenized_paded_documents)):
            for word_index in range(len(tokenized_paded_documents[0])):
                doc_word_embeddings[doc_index][word_index] = embedding_matrix[
                    tokenized_paded_documents[doc_index][word_index]
                ]

        return embedding_matrix, doc_word_embeddings

    def document_embeddings(self):

        tokenizer, tokenized_paded_documents = self.get_tokenized_elements()
        words, tfidf_vectors = self.get_tfidf_elements()
        embedding_matrix, doc_word_embeddings = self.document_word_embeddings()
        documentEmbeddings = np.zeros((len(tokenized_paded_documents), 300))
        for doc_index in range(len(doc_word_embeddings)):
            for word_index in range(len(words)):
                documentEmbeddings[doc_index] += (
                    embedding_matrix[tokenizer.word_index[words[word_index]]]
                    * tfidf_vectors[doc_index][word_index]
                )
        documentEmbeddings = documentEmbeddings / np.sum(tfidf_vectors, axis=1).reshape(
            -1, 1
        )

        return documentEmbeddings
