from turtle import window_width
import matplotlib
import streamlit as st
import pandas as pd
import nltk
import string 
import matplotlib.pyplot as plt
#import wordcloud
#from wordcloud import WordCloud 

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

import re
from nltk.tokenize import word_tokenize


import  pickle

pickle_in=open("SentimentIdentifier.pkl",'rb')
clf=pickle.load(pickle_in)

def predict(text):
    prediction=clf.predict(text)
    return prediction[0]

def preprocess(x):
  
  ## Lower casing 
  x=x.lower()

  ## Removing urls
  pattern=re.compile(r'https?://[\w./]+')
  x=pattern.sub(f'',x)

  ## Removing Punctutations
  for i in string.punctuation:
    x=x.replace(i,'')
  
  ## Tokenization
  x=word_tokenize(x)

  ## Removing Stopwords
  l=[]
  for i in x:
    if i not in stopwords.words('english'):
      l.append(i)
  
  return l

stemmer=PorterStemmer()

def root_words(x):
  l=[]
  for i in x:
    l.append(stemmer.stem(i))
  return l

#def Analyse(text):

    #wordcloud=WordCloud(width=1000,height=800,background_color='white').generate(tweet_pool)
    #plt.figure(figsize = (10, 6), facecolor = None)
    #plt.imshow(wordcloud)
    #plt.axis("off")
    #plt.tight_layout(pad = 0)
 
    #plt.show()


def main():

    html_title="""
    <div style="background-color: black; padding: 10px;">
    <h1 style="color: green; text-align: center;">Product review Sentiment</h1>
    </div>
    """
    st.markdown(html_title,unsafe_allow_html=True)

    st.header('About The Project :')
    st.write("""Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, 
    and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations. 
    Brands can use this data to measure the success of their products in an objective manner. 
    This Project predicts sentiment on electronic products of netizens.""")

    st.write("------------------------------------------------------------------------------------")

    st.subheader("Write Your review Below")
    text=st.text_area('',max_chars=200,height=200,placeholder="Enter Your Review")

    if st.button('Predict'):
        text=preprocess(text)

        text=root_words(text)

        ans=predict(text)

        if ans==1:
            negative="""
            <div style="padding: 10px;">
            <h1 style="color: red; text-align: center">Negative Review 😡</h1>
            </div>
            """
            st.markdown(negative,unsafe_allow_html=True)
        else:
            positive="""
            <div  style="padding: 10px">
            <h1 style="color: green; text-align: center">Positive Review 👍</h1>
            </div>"""

            st.markdown(positive,unsafe_allow_html=True)

if __name__=="__main__":
    main()
