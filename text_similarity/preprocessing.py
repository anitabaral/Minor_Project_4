import re
import string

import nltk
from nltk import word_tokenize
from unidecode import unidecode
from nltk.corpus import stopwords


def preprocess_corpus(sentences):
    """
    Takes news article as input and cleans them for better results while getting feature vectors

        Parameters:
            senteces (string): A news artice

        Returns:
            cleaned_corpus (string): A preprocessed news article with required words
    """

    new_sentence = ""
    sentences = re.sub(r"<\s*br\s*\/s*>", "", sentences)
    sentences = re.sub(r"\n>", " ", sentences)
    sentences = re.sub(r"\s+", " ", sentences)
    sentences = re.sub(r"\.+\s*", ".", sentences)
    # to substitute word  "who'll" with "who will"
    sentences = re.sub(r"who\'ll", "who will", sentences)
    # to substitute word "I'll", "i'll", "you'll", "she'll", "SHE'll" with "i will"
    sentences = re.sub(r"[IiyouYousheSHE]\'ll", "i will", sentences)
    # to substitute word  "wouldn't", "Wouldn't" with "would not"
    sentences = re.sub(r"[wW]ouldn\'t", "would not", sentences)
    # to substitute word  "mustn't", "Mustn't" with "must not"
    sentences = re.sub(r"[mM]mustn\'t", "must not", sentences)
    # to substitute word  "that's", "That's" with "would not"
    sentences = re.sub(r"[tT]hat\'s", "that is", sentences)
    sentences = sentences.replace(".", " ")
    for sentence in sentences:
        if sentence.isspace() or sentence.isalpha():
            new_sentence += sentence.lower()
    stopset = stopwords.words("english") + list(string.punctuation)
    corpus = " ".join(
        [word for word in word_tokenize(new_sentence) if word not in stopset]
    )
    cleaned_corpus = unidecode(corpus)

    return cleaned_corpus
