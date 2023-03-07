import matplotlib.pyplot as plt
import numpy as np
from FileLoader import *
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class Preprocessing(object):

    ## file should be a dataframe
    def __init__(self, df):
        self.__file = df

    ## attribute is a string
    def error_cleaning(self, attribute):
        self.__file[attribute] = self.__file[attribute].str.replace('(&amp)','')
        self.__file[attribute] = self.__file[attribute].str.replace('(\xa0)','')

    ## attribute is a string
    def sentence_normalizatio(self, attribute):
        self.__file[attribute] = self.__file[attribute].str.lower()

        self.__file[attribute] = self.__file[attribute].str.replace('won\'t', 'will not')
        self.__file[attribute] = self.__file[attribute].str.replace('can\'t', 'can not')
        self.__file[attribute] = self.__file[attribute].str.replace('n\'t', ' not')
        self.__file[attribute] = self.__file[attribute].str.replace('\'re', ' are')
        self.__file[attribute] = self.__file[attribute].str.replace('\'s', ' is')
        self.__file[attribute] = self.__file[attribute].str.replace('\'d', ' would')
        self.__file[attribute] = self.__file[attribute].str.replace('\'ll', ' will')
        self.__file[attribute] = self.__file[attribute].str.replace('\'t', ' not')
        self.__file[attribute] = self.__file[attribute].str.replace('\'ve', ' have')
        self.__file[attribute] = self.__file[attribute].str.replace('\'m', ' am')

        self.__file[attribute] = self.__file[attribute].str.replace('[\d+]', ' ')
        self.__file[attribute] = self.__file[attribute].str.replace('[^\w\s]', ' ')

        self.__file[attribute] = self.__file[attribute].apply(lambda x: nltk.word_tokenize(x))
        self.__file[attribute] = self.__file[attribute].apply(lambda x: [WordNetLemmatizer().lemmatize(word) for word in x])

        stop_words = stopwords.words('english')
        token = self.__file[attribute].apply(lambda x: [word for word in x if not word in stop_words])
        return token
