# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:34:17 2019

@author: Kem
"""
from Tweet_to_Emotion import twtToken, indexToken
import pandas as pd
from joblib import load
import numpy as np
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize

############################################################
#Method for detecting input language
def lang_ratios(twt):
    ratio = {}
    tok = wordpunct_tokenize(twt)
    words = [word.lower() for word in tok]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        ratio[language] = len(common_elements) # language "score"

    return ratio

def detect_lang(twt):
    ratio = lang_ratios(twt)
    max_ratio_lang = max(ratio, key=ratio.get)
    return max_ratio_lang

#######################################################
#load model and get user tweet input
    
clf = load('testcode.joblib')

word_list = load('word_list.joblib')

test_twt = str(input("Tweet: "))
while detect_lang(test_twt)!= 'english':
    print("Error! Detected input language: {}".format(detect_lang(test_twt)))
    test_twt = str(input("Tweet was not in English, input different tweet: "))


####################################################
#preprocessing input into format for model
    
tkn_twt = twtToken(test_twt)


in_tkn = indexToken(tkn_twt,word_list)
in_tkn = np.array(in_tkn).reshape((1,-1))

sentiment = ('anger', 'boredom', 'enthusiasm', 'fun', 'happiness','hate','love', 'relief','sadness', 'surprise', 'worry')

########################################################

t = clf.predict_proba(in_tkn)
d = {'Sentiment':sentiment, 'Probability':t[0]}

t_ems = pd.DataFrame(d)
t_ems = t_ems.sort_values('Probability', ascending = False)

########################################################
#display result

print("Primary Prediction: ")
print(t_ems.iloc[0,0])

print("Secondary Prediction: ")

print(t_ems.iloc[1,0])

