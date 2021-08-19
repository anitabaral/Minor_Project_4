import bs4
import pandas as pd
from pathlib import pathlib

class LoadFile:
    def __init__(self):
        pass
    
    def extract_reuters_news(self, path_file):
        
        file = open(path_file , 'r').read()
        soup = bs4.BeautifulSoup(file)
        news = [el.text for el in soup.find_all('reuters')] 

        return news

    def news_dataframe(self):

        file_path = Path('/content/drive/MyDrive/Leapfrog_internship/Project 4/reut2-021.sgm')
        if file_path.exists():
            news = extract_reuters_news(file_path)
        else:
            Print('-- There is error while reading the file. --')
        document = []
        for i, doc in enumerate(news):
            document.append(preprocess(doc))
        news_dataframe = pd.DataFrame(list(zip(news, document)), columns = ['Initial_corpus','Cleaned_corpus'])

        return news_dataframe