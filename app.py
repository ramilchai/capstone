from collections import defaultdict
from matplotlib import image
import streamlit as st
from PIL import Image
import pandas as pd
import pickle

from surprise import Reader, Dataset
from surprise.prediction_algorithms import SVD

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#######################################################
# import datasets 
#######################################################

df = pd.read_pickle('interact')
library = pd.read_pickle('lib_app')
rand_book = pd.read_pickle('random_app')

#######################################################
# Side Bar
#######################################################

st.sidebar.header('Start Here')
u_name = st.sidebar.text_input('What is your name?', 'Friends')
choice = st.sidebar.radio('How would you want to rate?', ['Choose for me', 'Choose my own'])
num = st.sidebar.slider(label='How many books?', min_value=1, max_value=10)


#######################################################
# Page Title
#######################################################

img_header = Image.open('images/logo.jpg')

st.image(img_header, use_column_width=True )
st.header('Books Recommender')

st.write("""

Hi, there! My name is Ramil. I create this web app to introduce fantasy book lovers to the awesome world of comic books.

***
""")


st.write(f'Hi, {u_name}! Please take some time to rate these following books')

if choice == 'Choose for me':


    #Sample Top Rating books
    lib = rand_book.sample(num).reset_index()
    entry = defaultdict(int)
    st.dataframe(lib)
    
    with st.form(key='CFM'):
        col1, col2 = st.beta_columns((1,2))

        
        for i in range(num):
            
            with col2:
                idx = lib.loc[i, 'book_id']
                title = lib.loc[i, 'title']
                avg = lib.loc[i,'average_rating']
                st.write(f"{title} has an average rating of {avg}")
            
                entry[int(idx)] = st.select_slider('How would you rate it?', ["Haven't Read", 1, 2 ,3 ,4, 5], key=f'select{i}')
            with col1:
                i = lib.loc[i,'image_url']
                response = requests.get(i)
                img = Image.open(BytesIO(response.content))
                st.image(img)


        button1 = st.form_submit_button("Submit")
    
    if button1:
        st.write(entry)


if choice == "Choose my own":
    book_name = library['']
    entry = defaultdict(int)
    with st.form(key='CMO'):
        col1, col2 = st.beta_columns((1,2))

        
        for i in range(num):
            
            with col2:
                idx = lib.loc[i, 'book_id']
                title = lib.loc[i, 'title']
                avg = lib.loc[i,'average_rating']
                st.write(f"{title} has an average rating of {avg}")
            
                entry[int(idx)] = st.select_slider('How would you rate it?', ["Haven't Read", 1, 2 ,3 ,4, 5], key=f'select{i}')
            with col1:
                i = lib.loc[i,'image_url']
                response = requests.get(i)
                img = Image.open(BytesIO(response.content))
                st.image(img)


        button2 = st.form_submit_button("Submit")
    
    if button2:
        st.write(entry)
        #with st.form(key='columns_in_CMO'):
            #cols = st.beta_columns(num)
            #for i, col in enumerate(cols):
                #col.selectbox(f'Make a Selection', ['click', 'or click'], key=i)
            
            #CMO_submit = st.form_submit_button(label='Submit')
   

#rec_idx = recommended_books(ranked_books, library, 10)



#######################################################
# SVD
#######################################################

user_rating = [{'user_id_num': 10000, 'book_id': 74043, 'rating': 3},
 {'user_id_num': 10000, 'book_id': 25938442, 'rating': 2},
 {'user_id_num': 10000, 'book_id': 18872831, 'rating': 4},
 {'user_id_num': 10000, 'book_id': 3090465, 'rating': 5},
 {'user_id_num': 10000, 'book_id': 17378508, 'rating': 4},
 {'user_id_num': 10000, 'book_id': 3586934, 'rating': 3},
 {'user_id_num': 10000, 'book_id': 7775569, 'rating': 1},
 {'user_id_num': 10000, 'book_id': 21839516, 'rating': 2}]

reader = Reader()
new_df = df.append(user_rating, ignore_index=True)
new_data = Dataset.load_from_df(new_df, reader)

svd_final_model = SVD(n_factors= 20, reg_all=0.02)
svd_final_model.fit(new_data.build_full_trainset())

list_of_books = []
for m_id in new_df['book_id'].unique():
    list_of_books.append( (m_id,svd_final_model.predict(10000,m_id)[3]))

ranked_books = sorted(list_of_books, key=lambda x:x[1], reverse=True)

def recommended_books(user_ratings, book_title_df, n):
    rec_list = []
    for idx, rec in enumerate(user_ratings):
        title = book_title_df.loc[book_title_df['book_id'] == int(rec[0])]
        #print('Recommendation #', idx+1, ': ', title['title'].values[0], '\n')
        st.write('Recommendation #', str(idx+1), ': ', title['title'].values[0], '\n')
        rec_list.append(title['book_id'].values[0])
        n -= 1
            
        i = title['image_url'].values[0]
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))
        #plt.figure()
        #print(plt.imshow(img))
        #st.pyplot(plt.imshow(img))
        st.image(img)
        if n == 0:
            return rec_list

