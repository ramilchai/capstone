# Comic Books Recommendation System  
## Based on Fantasy Book Preferences
![Goodreadslogo](./images/logo.jpg)
By Ramil Chaimongkolbutr

## Overview

Recommendation system has been an integral part of tech companies' success for years. It helps generate around 35% of Amazon's revenue; it increases 33% of Spotify users; it contributes to 75% of what users watch on Netflix; and it is accounted for 60% of video clicks on Youtube. Recommendation system is used in a variety of areas, with commonly recognized examples taking the form of playlist generators for video and music services, relatable product recommenders for online stores, or content recommenders for social media platforms, or personalized topics such as restaurants and online dating.

## Business Problems

Amazon Books are looking to expand their sale in fantasy book section. Their plan is not only limited to recommending books within a certain genres, but also introducing new books from a different genres. We are tasked to build up a recommendation system that allows us to recommend cross-genre books, which in our case, comic books, to a customer based on his/her preference in fantasy book.

## Data

Data was collected directly from Goodreads API in 2017 and made availble by [Mengting Wan et al.](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) online (please click the name for the link to download). For more infomation, please visit Mengting's [Github repository](https://github.com/MengtingWan/goodreads).

**Note**: the dataset is too large to be uploaded on github.  
Here are steps to acquire the dataset in order to make our codes work:
1. Download these following datasets:
    - "goodreads_books_comics_graphics.jason.gz" by this [link](https://drive.google.com/uc?id=1ICk5x0HXvXDp5Zt54CKPh5qz1HyUIn9m).
    - "goodreads_interactions_comics_graphic.json.gz" by this [link](https://drive.google.com/uc?id=1CCj-cQw_mJLMdvF_YYfQ7ibKA-dC_GA2).
    - "goodreads_books_fantasy_paranormal.json.gz" by this [link](https://drive.google.com/uc?id=1x8IudloezYEg6qDTPxuBkqGuQ3xIBKrt).
    - "goodreads_interactions_fantasy_paranormal.json.gz" by this [link](https://drive.google.com/uc?id=1EFHocJIh5nknbUMcz4LnrMEJkwW3Vk6h).
2.    Create a folder and name it "data", then move all the dataset into the folder.
3.    unzip all the data file. Make sure that all the data does not have .gz at the end of its name (gunzip command in gitbash can be useful in this case).  

The datasets we use in this project contain over 250000 fantasy books and over 85000 comic books. They also contain over 60 milion reviews from all over 100000 users. The rating reviews score between 1-5 stars (5 is "it is amazing!" to 1 is "Didn't like it at all"). The metadata also contains key features such as book_id, title, description, average rating, and image links.

## EDA
### Explore the Rating Counts
Rating Counts: Top 100 highest-rated books vs The Rest 

![rating_count](./images/Top100.png)

From the graph, we can see that the higher rated books trend to attach more reviewers. In top 100 highest-rated books, it has almost a half as many reviews as the rest of 258485 books. Let's get to know which one has the most rating counts.

### Word Count Distribution
We want to gain some insight from book descriptions. We plan to use this feature to find similarity between two datasets. Since we are working with the fantasy book dataset, we start by taking a look on its word count distribution first.

![Word_count](./images/word_counts_new.png)

The average word count is around 200 words.

### Part-of-speech tags Distribution in book descriptions
Let's check out the distribution of its part-of-speech. This section we will use the help from Textblob library. Textblob library is an open-source python library for processing textual data. It performs different operations on textual data such as noun phrase extraction, sentiment analysis, classification, translation, etc.

![pos](./images/POS.png)

### Bigram and Trigram Distribution
Let's take a look on the distribution of bigram and trigram.

![Bigram](./images/bigram_new.png)
![Trigram](./images/trigram.png)

### Exploring the interaction dataset
Distribution of ratings in our interaction dataset.

![rating_users](./images/rating_dist_new.png)

The distribution of rating in our dataset is left-skewed. Majority of users trend to rate books are quite optimistic, meaning that they trend to rate books with high regards.

Box plot for user interactions.

![Box_plots](./images/Box_users.png)

As you can see, the medium is 26 reviews per user while the mean is around 57 review per user. The large gap is a result of having some outliers with extremely high numbers of interactions. For example, there is one user rates the total of 733 books; there are a few users rate above 500 books.

## Methodology

We begin our methodology with filtering data down to more manageable size. We remove unrelated and unsavory genres such as 'Musical', 'Movie','OVA', 'Special', 'ONA', and 'Hentai.' We choose to build our recommendation system using collaborative filtering algorithm. There are two libraries that provide useful built-in machine learning models: ALS from PySpark; SVD from Surprise. These are two models that we rely on heavily.  


## Results

From ALS model, RMSE is around 5.6.
From SVD model, RMSE is around 3.1, which is better judging by the number.

# Example of results from SVD model:
We add a new user which some preference.
![table1](./imgs/input.png)

Here are 5 examples out of the total 10 recommended anime.
1. Code Geass: Hangyaku no Lelouch R2
2. Major S6
3. Death Note
4. One Punch Man
5. Black Lagoon: The Second Barrage

From 10 recommended results, we have found that 4 of them are already seen and liked; 2 of them are recommended by other services; 1 is new but recommended by friends; and 3 of them are never heard but promising.

## Conclusions

- SVD model does the best job with 3.1 RMSE.
- We are satisfied with the results so far.

## Next Steps

In order to improve results, we might need to break down the data even more. Some anime in the dataset has multiple seasons. Combining them might give a better RMSE score. 
 
## For More Information

See full analysis in [Jupyter Notebook](./phase_4_code.ipynb) or check out this [presentation](./phase_4_slidedeck.pdf). 
For additional info, please contact:  
Ramil Chaimongkolbutr at [ramil.ming@flatironschool.com](mailto:ramil.ming@flatironschool.com)  
George Ferre at [georgeaferre@flatironschool.com](mailto:georgeaferre@flatironschool.com)  
Aaron Cherry at [cherrya050@flatironschool.com](mailto:cherrya050@flatironschool.com)

![Ending](./imgs/endinggif.gif)

## Repository Structure

```
├── data
├── imgs
├── notebooks
├── README.md
├── phase_4_slidedeck.pdf
└── phase_4_code.ipynb
```
