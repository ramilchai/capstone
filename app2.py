from collections import defaultdict
#from matplotlib import image
import streamlit as st
from PIL import Image
import pandas as pd
#import pickle

from surprise import Reader, Dataset
from surprise.prediction_algorithms import SVD

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

#nltk.data.path.append('./corpora')

import re
#import string
#import random
import requests
from io import BytesIO
#import matplotlib.pyplot as plt

#######################################################
# Functions
#######################################################
@st.cache(allow_output_mutation=True)
def openFile(name):
    return pd.read_pickle(name)


def show():
    img_header = Image.open('images/logo.jpg')

    st.image(img_header, use_column_width=True )
    st.header('Books Recommender')

    st.write("""Hi, there! My name is Ramil. I create this web app to introduce fantasy book lovers to the awesome world of comic books.

***
    """)

def transformDict(ent, userID=10000):
    arr = []
    for key, value in ent.items():
        if value == "Haven't Read":
            continue
        else:
            n = {'user_id_num': userID, 'book_id': key, 'rating': value}
            arr.append(n)
    
    return arr

def getRec(user_rating, data):
    reader = Reader()
    new_df = data.append(user_rating, ignore_index=True)
    new_data = Dataset.load_from_df(new_df, reader)

    svd_final_model = SVD(n_factors= 20, reg_all=0.02)
    svd_final_model.fit(new_data.build_full_trainset())

    list_of_books = []
    for m_id in new_df['book_id'].unique():
        list_of_books.append( (m_id,svd_final_model.predict(10000,m_id)[3]))

    ranked_books = sorted(list_of_books, key=lambda x:x[1], reverse=True)
    return ranked_books

def recommended_books(user_ratings, book_title_df, n):
    rec_list = []
    for idx, rec in enumerate(user_ratings):
        title = book_title_df.loc[book_title_df['book_id'] == int(rec[0])]
        
        st.write('Recommendation #', str(idx+1), ': ', title['title'].values[0], '\n')
        rec_list.append(title['book_id'].values[0])
        n -= 1
            
        i = title['image_url'].values[0]
        response = requests.get(i)
        img = Image.open(BytesIO(response.content)).convert('RGB')
 
        st.image(img)
        st.write('***')
        if n == 0:
            return rec_list

def recommended_comics(arr, comics_df):
        for idx, rec in enumerate(arr):
            title = comics_df.loc[comics_df['book_id'] == int(rec)]
            
            st.write('Recommendation #', str(idx+1), ': ', title['title'].values[0], '\n')

            i = title['image_url'].values[0]
            response = requests.get(i)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            st.image(img)
            st.write('***')

# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)
# Function for converting into lower case
def make_lower_case(text):
    return text.lower()
# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text
#Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

#######################################################
# Header 
#######################################################

if __name__=="__main__":
    show()

    df = openFile('interact')
    library = openFile('lib_app')
    comics_lib = openFile('comics_lib')
    rand_book = openFile('random_app')
    df_comics = openFile('comics_cleaned_description')

#######################################################
# Sidebar
#######################################################

st.sidebar.header('Start Here')
u_name = st.sidebar.text_input('What is your name?', 'Friends')
choice = st.sidebar.radio('How would you want to rate?', ['Choose for me', 'Choose my own'])

if choice == "Choose for me":   
    st.sidebar.header(f'Hi, {u_name}!')
    num = st.sidebar.slider(label='How many books?', min_value=1, max_value=10, key="CFM_num")
    lib = rand_book.sample(num).reset_index()
    

else:
    st.sidebar.header(f'Hey, {u_name}! Let"s rate the books!')
    book_name = list(set(library['title'].values))
    select = st.sidebar.multiselect('Book Name(s)', book_name)
    num = len(select)
    lib = library[library['title'].isin(select)].drop_duplicates(subset=['title']).reset_index()

#######################################################
# Page
#######################################################

entry = defaultdict(int)
with st.form(key='user_submit_rating'):

    for i in range(num):
            
        url = lib.loc[i,'image_url']
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(img)
        

        idx = lib.loc[i, 'book_id']
        title = lib.loc[i, 'title']
                
        if choice == "Choose for me":
            avg = lib.loc[i,'average_rating']
            st.write(f"{title} has an average rating of {avg}")
        else:
            st.write(f"{title}")
            
        entry[int(idx)] = st.select_slider('How would you rate it?', ["Haven't Read", 1, 2 ,3 ,4, 5], key=f'select{i}')

        st.write('***')
            
            
    user_comics = st.checkbox('Comics')
    button = st.form_submit_button("Recommend Me!")

if button:
    user_rating = transformDict(entry)
    st.header(f'This is a fantasy book recommendation for {u_name}')
    rb = getRec(user_rating, df)
    rec_idx = recommended_books(rb, library, 10)
    
    if user_comics:
        st.header(f'This is a comic book recommendation for {u_name}')
        rec_book_df = comics_lib[comics_lib['book_id'].isin(rec_idx)]
        rec_book_df['cleaned_des'] = rec_book_df['description'].apply(_removeNonAscii)
        rec_book_df['cleaned_des'] = rec_book_df['cleaned_des'].apply(func = make_lower_case)
        rec_book_df['cleaned_des'] = rec_book_df['cleaned_des'].apply(func = remove_stop_words)
        rec_book_df['cleaned_des'] = rec_book_df['cleaned_des'].apply(func = remove_punctuation) 
        rec_book_df['cleaned_des'] = rec_book_df['cleaned_des'].apply(func = remove_html)

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df = 1, stop_words='english')
        tfidf_matrix = tf.fit_transform(rec_book_df['cleaned_des'])
        comics_matrix = tf.transform(df_comics['cleaned_des'])

        cos_sim = cosine_similarity(comics_matrix, tfidf_matrix)
        cos_sim_df = pd.DataFrame(cos_sim, index=df_comics['book_id'] , columns=rec_book_df['book_id'])
        rec_comic_idx = [cos_sim_df[x].idxmax() for x in cos_sim_df.columns]

        recommended_comics(rec_comic_idx, df_comics)