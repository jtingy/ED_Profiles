"# ED_Profiles" 

Original data: https://www.kaggle.com/datasets/jabenitez88/eating-disorders-tweets?select=tweets.csv

Data used for analyses: https://kaggle.com/datasets/455e5871c171a36d78df3f645493902c56a82e93a615bf0eb7968382fba74a69

To Replicate Analyses:
1. Run processData.py - Requires 'tweets.csv' (from Kaggle link above), Output is 'ed_tweets_labeled_full.csv'
2. Run classify_and_label.py - Requires 'sample.csv' (in data.zip) and 'ed_tweets_labeled_full.csv', Output is 'ed_tweets_final.csv'
3. Run Ranalysis.Rmd - Requires 'ed_tweets_final.csv'
