import yaml

from text_similarity import LoadFile, PretrainedModels, preprocess_corpus, Embeddings, SimilarDocuments, most_similar

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)

def main():

    # Embeddings(file_paths['data_loc'], file_paths['model_loc'])
    SimilarDocuments(file_paths['data_loc'], file_paths['model_loc'])
    print(most_similar(file_paths['data_loc'],  file_paths['model_loc'], 0,'Cosine Similarity'))
    # most_similar(0,'Euclidean Distance')

    

main()