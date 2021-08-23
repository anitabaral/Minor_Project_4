<h3> Most simliar news articles. </h3>

<h4> The project finds the top  5 most similar articles from a given news article dataset. </h4>

  <h5> Implementation and componentt details: </h5>
  
  > Load the gensim model: load_model.py 

  > Load the news article dataset: load_data.py 

  > Preprocess and clean the data: preprocessing.py 

  > Get features vectors using gensim model: embedding.py 

  > Calculating cosine similarities and euclidean distances: document_similarity.py 

  > Printing top 5 similar news articles: view_similar_articles.py 

<h4> Steps to run the repo </h4>

- Clone the repository: git clone  https://github.com/anitabaral/Minor_Project_4.git
- Install pipenv

<h5>1. pipenv setup </h5>

  - Go to the project folder 
  - Open terminal 
  - Run the command: pipenv shell 
  - Install all the components specified in the Pipfile.

 <h5>2. Model setup </h5>
 
 -  Get the pretrained vectors from here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
 -  Save the content inside a folder name 'model'

<h5>3. Dataset setup </h5>

- Download the tar file from the link: https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/
- Extract **reut2-021.sgm** file inside a 'data'

<h5>4. To run the code </h5>
  - python app.py

<h4>Output: </h4>
Top 5 similar articles to the article specified by the index.