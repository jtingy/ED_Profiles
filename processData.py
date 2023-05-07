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
from sklearn.metrics import accuracy_score
np.random.seed(2001)

##Tfidf_vect = TfidfVectorizer(max_features=5000)
# Tfidf_vect.fit(Corpus['text_final'])
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)

def main():
    ed_df = pd.read_csv('tweets.csv', encoding="ISO-8859-1")
    ## Delete duplicates
    ed_df = ed_df.drop_duplicates(subset=["author"])


    print("duplicates deleted")
    print(len(ed_df))
    
    # Preprocess and label
    ed_df = prepStep(ed_df)

    # Get rid of invalid observations
    ed_df = ed_df[ed_df["final_desc"] != "[]"]
    ed_df = ed_df[ed_df["final_desc"] != ""]
    ed_df = ed_df[ed_df["final_desc"] != "['none']"]

    # Save to new csv
    ed_df.to_csv("ed_tweets_labeled_full.csv", index=False)
    print(ed_df)
    print(len(ed_df))



## https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
def prepStep(df):
    ## Preprocessing
    df['desc_working'] = df['description']
    # Step - a : Remove blank rows if any.
    df['desc_working'].dropna(inplace=True)
    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    df['desc_working'] = [str(entry).lower() for entry in df['desc_working']]
    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    df['desc_working']= [word_tokenize(entry) for entry in df['desc_working']]
    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    ## Make a list of final_words list - declare empty list
    desc_list = []
    desc_col = []
    #number_list = list(df['id_num'])
    for index,entry in enumerate(df['desc_working']):
        #
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        desc_col.append(str(Final_words))
        desc_list.append(" "+" ".join(Final_words)+" ")
        #print(len(df))

    df = df.drop('desc_working', axis='columns')
    df["final_desc"] = desc_col
    #print(desc_list)

    
    ## Labeling
    df = labelText(df, desc_list)

    return df

def labelText(df, desc_str_list):
    ## Dictionary to label as pro-ed
    label_pro = ['thinspo','meanspo','fatspo','best anorexic', 'edtwt','proana', ' proed ', ' pro ed ', ' gw ', ' ugw ', ' cw ', ' ed ', 'eating disorder', 'eat disorder', '3dtwt']
    label_anti = ['pro recovery']
    weight_words = ['lbs', 'kgs', ' gw ', ' ugw ', ' cw ', ' bmi ', 'kg', 'lb']
    tw_words = [' tw ', ' dni ', 'trigger warning', ' dfi ', "do not interact", "do interact", "interact"]
    sh_words = [' sh ', 'self harm', 'self hate']
    vent_words = [' vent ']

    label_col = [0]*len(df)
    prorec_col = [0]*len(df)
    weight_col = [0]*len(df)
    tw_col = [0]*len(df)
    sh_col = [0]*len(df)
    vent_col = [0]*len(df)
    for index, desc  in enumerate(desc_str_list):
        ## Go through list of phrases that come up in ed profiles to label profile as pro-ed
        for j in label_pro:
            ## If phrase is in the string, label pro = 1
            if j in desc:
                label_col[index] = 1
                
                break
        
        for h in weight_words:
            if h in desc:
                weight_col[index] = 1
                break

        for k in tw_words:
            if k in desc:
                tw_col[index] = 1
                break
        
        for l in sh_words:
            if l in desc:
                sh_col[index] = 1
                break

        for m in vent_words:
            if m in desc:
                vent_col[index] = 1
                break

        for n in label_anti:
            #print(n)
            if n in desc:
                #print(desc)
                prorec_col[index] = 1
                break

    
    df["desc_label"] = label_col
    df["pro_recovery"] = prorec_col
    df["weight_label"] = weight_col
    df["tw_label"] = tw_col
    df["sh_label"] = sh_col
    df["vent_label"] = vent_col
    return df





## Global call to run code
main()