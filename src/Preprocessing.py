import re
import string

import nltk
from nltk import word_tokenize
from unidecode import unidecode
from nltk.corpus import stopwords


class Preprocessing:
    def __init__(self):
        pass

    def preprocess_corpus(self, sentences):

        new_sentence = ""
        sentences = re.sub(r"<\s*br\s*\/s*>", "", sentences)
        sentences = re.sub(r"\n>", " ", sentences)
        sentences = re.sub(r"\s+", " ", sentences)
        sentences = re.sub(r"\.+\s*", ".", sentences)
        sentences = re.sub(r"who\'ll", "who will", sentences)
        sentences = re.sub(r"[IiyouYousheSHE]\'ll", "i will", sentences)
        sentences = re.sub(r"[wW]ouldn\'t", "would not", sentences)
        sentences = re.sub(r"[mM]mustn\'t", "must not", sentences)
        sentences = re.sub(r"[tT]hat\'s", "that is", sentences)
        # sentences = re.sub(r'oct', ' ', sentences)
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
