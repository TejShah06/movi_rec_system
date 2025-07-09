from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load and preprocess data (from notebook logic)
movie = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')
movie = movie.merge(credits, on='title')
movie = movie[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language']]
movie.dropna(inplace=True)

def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

def convert_c(text):
    l = []
    c = 0
    for i in ast.literal_eval(text):
        if c < 5:
            l.append(i['name'])
        c += 1
    return l

def convert_cr(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l

movie['genres'] = movie['genres'].apply(convert)
movie['keywords'] = movie['keywords'].apply(convert)
movie['cast'] = movie['cast'].apply(convert_c)
movie['crew'] = movie['crew'].apply(convert_cr)
movie['overview'] = movie['overview'].apply(lambda x: x.split())
movie['genres'] = movie['genres'].apply(remove_space)
movie['keywords'] = movie['keywords'].apply(remove_space)
movie['cast'] = movie['cast'].apply(remove_space)
movie['crew'] = movie['crew'].apply(remove_space)
movie['tags'] = movie['overview'] + movie['genres'] + movie['keywords'] + movie['cast'] + movie['crew']
movies = movie[['movie_id', 'title', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x: " ".join(map(str, x)))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()
def steam(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

movies['tags'] = movies['tags'].apply(steam)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
sim = cosine_similarity(vector)

def recommend(movie_title):
    index_list = movies[movies['title'].str.lower() == movie_title.lower()].index
    if index_list.empty:
        return []
    index = index_list[0]
    distances = sorted(list(enumerate(sim[index])), reverse=True, key=lambda x: x[1])
    recommended = []
    for i in distances[1:6]:
        recommended.append(movies.iloc[i[0]]['title'])
    return recommended

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form['movie']
        recommendations = recommend(movie_title)
        return render_template('result.html', movie=movie_title, recommendations=recommendations)
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True) 