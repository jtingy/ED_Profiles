"# ED_Profiles" 

Data: https://www.kaggle.com/datasets/jabenitez88/eating-disorders-tweets?select=tweets.csv

To Replicate Analyses:
1. Run processData.py - Requires 'tweets.csv' (from Kaggle link above), Output is 'ed_tweets_labeled_full.csv'
2. Run classify_and_label.py - Requires 'sample.csv' (in data.zip) and 'ed_tweets_labeled_full.csv', Output is 'ed_tweets_final.csv'
3. Run Ranalysis.Rmd - Requires 'ed_tweets_final.csv'
