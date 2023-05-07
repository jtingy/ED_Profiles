import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(2001)

ed_df = pd.read_csv('ed_tweets_labeled_full.csv', encoding="ISO-8859-1")
ed_df["followers"] = ed_df["followers"].astype('int')
ed_df["stdFollowers"] = ( ed_df["followers"] - np.mean(ed_df["followers"]) )/np.std(ed_df["followers"])
ed_df = ed_df[ed_df["stdFollowers"] <= 3] 
ed_df["log_followers"] = (ed_df["followers"]+1).apply(np.log)
df = pd.read_csv('sample.csv', encoding="ISO-8859-1")
df["followers"] = df["followers"].astype('int')
df["stdFollowers"] = ( df["followers"] - np.mean(df["followers"]) )/np.std(df["followers"])
df = df[df["stdFollowers"] <= 3] 
df["log_followers"] = (df["followers"]+1).apply(np.log)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['final_desc'],df['desc_label'],test_size=0.4)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(ed_df['final_desc'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
cm = confusion_matrix(Test_Y, predictions_SVM)
print(cm)
desc_Tfidf = Tfidf_vect.transform(ed_df['final_desc'][ed_df['desc_label'] == 1])
print(len(ed_df['final_desc'][ed_df['desc_label'] == 1]))
#print(desc_Tfidf)
predict_desc = SVM.predict(desc_Tfidf)
print(len(predict_desc))
x = len(predict_desc[predict_desc == 1])/len(predict_desc)
print(ed_df['desc_label'][ed_df['desc_label'] == 1]-predict_desc)
ed_df['desc_label'][ed_df['desc_label'] == 1] = predict_desc
ed_df.to_csv("ed_tweets_final.csv", index=False)