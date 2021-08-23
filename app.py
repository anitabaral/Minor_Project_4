import yaml

from text_similarity import LoadFile, PretrainedModels, preprocess_corpus, Embeddings, SimilarDocuments, most_similar

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)

if __name__ == "__main__":

    SimilarDocuments(file_paths['data_loc'], file_paths['model_loc'])
    id = input('Enter the document id.')
    print(most_similar(file_paths['data_loc'],  file_paths['model_loc'], id,'Cosine Similarity'))

    

