import bs4
import pandas as pd
from pathlib import Path

from .preprocessing import preprocess_corpus


class LoadFile:
    def __init__(self, data_path):

        self.data_path = Path(data_path)

    def extract_reuters_news(self, file_path):
        """
        Extract only the body of articles from the html file

            Parameters:
                file_path (str): The path of the news articles sgm file.

            Returns:
                news_df (list): Return the list of news articles.
        """

        with open(file_path, "r") as file:
            soup = bs4.BeautifulSoup(file, features="html.parser")
        news = [reuter.text for reuter in soup.find_all("reuters")]

        return news

    def news_dataframe(self):
        """
        Makes the dataframe of the preprocessed reuters

            Parameters:
                None

            Return:
                news_df (object): Dataframe consisting of 578 news articles consisting both initial and cleaned corpus
        """

        if self.data_path.exists():
            news = self.extract_reuters_news(self.data_path)
        else:
            raise ValueError("Error while reading the file")
        document = []
        for index, doc in enumerate(news):
            document.append(preprocess_corpus(doc))
        news_df = pd.DataFrame(
            list(zip(news, document)), columns=["Initial_corpus", "Cleaned_corpus"]
        )

        return news_df
