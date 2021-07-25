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

@st.cache
def openFile(name):
    return pd.read_pickle(name)

df = openFile('interact')
library = openFile('lib_app')
rand_book = openFile('random_app')

#######################################################
# Side Bar
#######################################################

st.sidebar.header('Start Here')
u_name = st.sidebar.text_input('What is your name?', 'Friends')
choice = st.sidebar.radio('How would you want to rate?', ['Choose for me', 'Choose my own'])


if choice == "Choose for me":
    with st.sidebar.form(key='user_CFM'):
        st.header(f'Hi, {u_name}!')
        num = st.slider(label='How many books?', min_value=1, max_value=10, key="CFM_num")
        lib = rand_book.sample(num).reset_index()
        submit = st.form_submit_button('Submit')

else:
    with st.sidebar.form(key='user_CMO'):
        st.header(f'Hey, {u_name}! Let"s rate the books!')
        book_name = list(set(library['title'].values))
        select = st.multiselect('Book Name(s)', book_name)
        num = len(select)
        lib = library[library['title'].isin(select)].drop_duplicates(subset=['title']).reset_index()
        submit = st.form_submit_button('Submit')

#######################################################
# Page 
#######################################################

img_header = Image.open('images/logo.jpg')

st.image(img_header, use_column_width=True )
st.header('Books Recommender')

st.write("""

Hi, there! My name is Ramil. I create this web app to introduce fantasy book lovers to the awesome world of comic books.

***
""")

st.write(f'{u_name}! Please take some time to rate these following books?')


if "entry" not in st.session_state:
    st.session_state.entry = defaultdict(int)

if submit:
    st.dataframe(lib)
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
            
            

        button = st.form_submit_button("Submit")





  

    


   

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

