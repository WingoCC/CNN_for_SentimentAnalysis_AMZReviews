from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from FileLoader import *
import re
import nltk
from collections import Counter
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score

pd.set_option('display.width', 150)
pd.set_option('display.max_colwidth', 150)

training_data = FileLoader("Covid_train_data.csv")
df_training = training_data.read_file()

testing_data = FileLoader("Covid_test_data.csv")
df_testing = testing_data.read_file()

print(df_training.info())
print(df_testing.info())

df_training["Sentiment"].value_counts().plot(kind='bar')
plt.xlabel("Sentiment")
plt.ylabel("Counts")
plt.title("Proportion of sentiments")
plt.show()

def sentiment_extraction(df, column, label):
    df = df.loc[df[column] == label]
    return df


def sentence_normalization(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
    # ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    text = text.lower()

    text = text.replace('won\'t', 'will not')
    text = text.replace('can\'t', 'can not')
    text = text.replace('n\'t', ' not')
    text = text.replace('\'re', ' are')
    text = text.replace('\'s', ' is')
    text = text.replace('\'d', ' would')
    text = text.replace('\'ll', ' will')
    text = text.replace('\'t', ' not')
    text = text.replace('\'ve', ' have')
    text = text.replace('\'m', ' am')

    text = re.sub(r'\d+', '', text)
    text = text.replace('[^\w\s]', ' ')

    token = nltk.word_tokenize(text)
    token = [WordNetLemmatizer().lemmatize(word) for word in token]

    stop_words = stopwords.words('english')
    token = [word for word in token if not word in stop_words]
    return token

def get_word_frequency(token):
    frequency_dict = nltk.FreqDist(token)
    return frequency_dict


def word_frequency(token):
    total_dict_counter = Counter({})
    for i in range(len(token)):
        sub_dict_counter = Counter(token[i])
        total_dict_counter = total_dict_counter + sub_dict_counter

    total_dict = dict(total_dict_counter)
    total_dict = dict([(k, v) for k, v in total_dict.items() if len(k) > 2])
    return total_dict


def form_a_tfidf_matrix(token):
   connect = ' '
   for i in range(len(token)):
      token[i] = connect.join(token[i])

   tv = TfidfVectorizer(min_df=0.1, max_df=0.4, use_idf=True)
   tv_matrix = tv.fit_transform(token)
   tv_matrix = tv_matrix.toarray()
   vocab = tv.get_feature_names()
   matrix_result = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
   return matrix_result, vocab


df_training["OriginalTweet"] = df_training["OriginalTweet"].apply(sentence_normalization)
df_testing["OriginalTweet"] = df_testing["OriginalTweet"].apply(sentence_normalization)

# print("Training Data after normalization:\n", df_training["OriginalTweet"])
# print("Test Data after normalization:\n", df_testing["OriginalTweet"])
#
# df_training_extremely_positive = sentiment_extraction(df_training, "Sentiment", "Extremely Positive")
# df_training_extremely_positive = df_training_extremely_positive.reset_index(drop=True)
# print("Categorised data frame for \"Extremely Positive\"")
# print(df_training_extremely_positive)
# df_training_positive = sentiment_extraction(df_training, "Sentiment", "Positive")
# df_training_positive = df_training_positive.reset_index(drop=True)
# print("Categorised data frame for \"Positive\"")
# print(df_training_positive)
# df_training_neutral = sentiment_extraction(df_training, "Sentiment", "Neutral")
# df_training_neutral = df_training_neutral.reset_index(drop=True)
# print("Categorised data frame for \"Neutral\"")
# print(df_training_neutral)
# df_training_negative = sentiment_extraction(df_training, "Sentiment", "Negative")
# df_training_negative = df_training_negative.reset_index(drop=True)
# print("Categorised data frame for \"Negative\"")
# print(df_training_negative)
# df_training_extremely_negative = sentiment_extraction(df_training, "Sentiment", "Extremely Negative")
# df_training_extremely_negative = df_training_extremely_negative.reset_index(drop=True)
# print("Categorised data frame for \"Extremely Negative\"")
# print(df_training_extremely_negative)
#
# # generate a word cloud for Extremely Positive
# df_frequency_extremely_positive = df_training_extremely_positive
# for i in range(len(df_training_extremely_positive)):
#     df_frequency_extremely_positive['OriginalTweet'][i] = get_word_frequency(df_training_extremely_positive['OriginalTweet'][i])
#
# frequency_extremely_positive = word_frequency(df_frequency_extremely_positive['OriginalTweet'])
# frequency_dict_extremely_positive = nltk.FreqDist(frequency_extremely_positive)
# wcloud_extremely_positive = WordCloud().generate_from_frequencies(frequency_dict_extremely_positive)
# plt.imshow(wcloud_extremely_positive, interpolation='bilinear')
# plt.axis("off")
# (-0.5, 399.5, 199.5, -0.5)
# plt.show()
#
# # generate a word cloud for Positive
# df_frequency_positive = df_training_positive
# for i in range(len(df_training_positive)):
#     df_frequency_positive['OriginalTweet'][i] = get_word_frequency(df_training_positive['OriginalTweet'][i])
#
# frequency_positive = word_frequency(df_frequency_positive['OriginalTweet'])
# frequency_dict_positive = nltk.FreqDist(frequency_positive)
# wcloud_positive = WordCloud().generate_from_frequencies(frequency_dict_positive)
# plt.imshow(wcloud_positive, interpolation='bilinear')
# plt.axis("off")
# (-0.5, 399.5, 199.5, -0.5)
# plt.show()
#
# # generate a word cloud for Neutral
# df_frequency_neutral = df_training_neutral
# for i in range(len(df_training_neutral)):
#     df_frequency_neutral['OriginalTweet'][i] = get_word_frequency(df_training_neutral['OriginalTweet'][i])
#
# frequency_neutral = word_frequency(df_frequency_neutral['OriginalTweet'])
# frequency_dict_neutral = nltk.FreqDist(frequency_neutral)
# wcloud_neutral = WordCloud().generate_from_frequencies(frequency_dict_neutral)
# plt.imshow(wcloud_neutral, interpolation='bilinear')
# plt.axis("off")
# (-0.5, 399.5, 199.5, -0.5)
# plt.show()
#
# # generate a word cloud for Negative
# df_frequency_negative = df_training_negative
# for i in range(len(df_training_negative)):
#     df_frequency_negative['OriginalTweet'][i] = get_word_frequency(df_training_negative['OriginalTweet'][i])
#
# frequency_negative = word_frequency(df_frequency_negative['OriginalTweet'])
# frequency_dict_negative = nltk.FreqDist(frequency_negative)
# wcloud_negative = WordCloud().generate_from_frequencies(frequency_dict_negative)
# plt.imshow(wcloud_negative, interpolation='bilinear')
# plt.axis("off")
# (-0.5, 399.5, 199.5, -0.5)
# plt.show()
#
# # generate a word cloud for Extremely Negative
# df_frequency_extremely_negative = df_training_extremely_negative
# for i in range(len(df_training_extremely_negative)):
#     df_frequency_extremely_negative['OriginalTweet'][i] = get_word_frequency(df_training_extremely_negative['OriginalTweet'][i])
#
# frequency_extremely_negative = word_frequency(df_frequency_extremely_negative['OriginalTweet'])
# frequency_dict_extremely_negative = nltk.FreqDist(frequency_extremely_negative)
# wcloud_extremely_negative = WordCloud().generate_from_frequencies(frequency_dict_extremely_negative)
# plt.imshow(wcloud_extremely_negative, interpolation='bilinear')
# plt.axis("off")
# (-0.5, 399.5, 199.5, -0.5)
# plt.show()

matrix_train, vocab = form_a_tfidf_matrix(df_training["OriginalTweet"])
print("TFIDF matrix is constructed!\n")
print("Words included in the matrix are:")
print(vocab,'\n')
print("TFIDF matrix:")
print(matrix_train)


data_train = np.array(df_training['OriginalTweet'])
label_train = np.array(df_training['Sentiment'])

df_testing['OriginalTweet'] = df_testing['OriginalTweet'].apply(lambda x : ' '.join(x))
data_test = np.array(df_testing['OriginalTweet'])
label_test = np.array(df_testing['Sentiment'])

cv = CountVectorizer(binary=False, min_df=0.0, max_df=0.3)
cv_train_features = cv.fit_transform(data_train)
cv_test_features = cv.transform(data_test)


# building a SVM
svm = LinearSVC(penalty='l2', C=1, random_state=42, max_iter=2000)
svm.fit(cv_train_features, label_train)
svm_bow_cv_scores = cross_val_score(svm, cv_train_features, label_train, cv=5)
svm_bow_cv_mean_score = np.mean(svm_bow_cv_scores)

print('CV Accuracy (5-fold):', svm_bow_cv_scores)
print('Mean CV Accuracy:', svm_bow_cv_mean_score)
svm_bow_test_score = svm.score(cv_test_features, label_test)
print('Test Accuracy:', svm_bow_test_score)

category = {'Extremely Positive': 0, 'Positive': 1, 'Neutral': 2, 'Negative': 3, 'Extremely Negative': 4}

tfidftransformer = TfidfTransformer()
tfidf_train = tfidftransformer.fit_transform(cv.fit_transform(data_train))
weight_train = tfidf_train.toarray()
label_train = np.vectorize(category.get)(label_train)

tfidf_test = tfidftransformer.transform(cv.transform(data_test))
weight_test = tfidf_test.toarray()
label_test = np.vectorize(category.get)(label_test)
print(weight_train)

# temp = pd.DataFrame(data=weight_train[0:, 0:], index=range(weight_train.shape[0]), columns=range(weight_train.shape[1]))
# sns.heatmap(temp.corr(), annot=True)

xgb_train = xgb.DMatrix(tfidf_train, label=label_train)
xgb_test = xgb.DMatrix(tfidf_test, label=label_test)

param = { 'max_depth' : 6, 'eta': 0.3, 'objective': 'multi:softmax', 'num_class': 5}
epochs= 1000
xgb_model = xgb.train(param, xgb_train, epochs)

prediction = xgb_model.predict(xgb_test)
print('Accuracy score for XGB', accuracy_score(label_test, prediction))