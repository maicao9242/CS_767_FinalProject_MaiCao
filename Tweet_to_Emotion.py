import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


#load csv file into a variable and label sentiments into integers for classification

x = pd.read_csv('TagEmotion.csv')
#
sentiment = {'empty':0,'anger':1, 'boredom':2, 'enthusiasm':3, 'fun':4, 'happiness':5,'hate':6,'love':7, 'neutral':8, 'relief':9,'sadness':10, 'surprise':11, 'worry':12}

x=x.drop(x[x.sentiment == 'neutral'].index) #drop any data point that has neutral sentiment
x = x.drop(x[x.sentiment == 'empty'].index) #drop any data point that has empty sentiment


#split data into training data and validating data:
train = x.sample(frac = 0.7, random_state = 10)
valid = x.drop(train.index)


    
def twtToken(tweet): #method to tokenize tweet and remove stop words from token list
    tknzr = TweetTokenizer()
    
    words = tknzr.tokenize(tweet)
    
    stop = set(stopwords.words("english"))
    filtered_twt = []
    for w in words:
        if w not in stop:
            filtered_twt.append(w)
    
    return filtered_twt


def make_word_list(tweet):
    word_list = [] #a full list of words in whole data - for weight, removing any stop words from list
    stop = set(stopwords.words("english"))
    for x in tweet:
        for y in x:
            if (y not in word_list and y not in stop and '@' not in y ):
                word_list.append(y)
    word_list.sort()
    return word_list
    
    
def indexToken(tweet,words):
    tweet_word = []
    for token in words:
        word_count = 0
        if token in tweet:
            word_count += 1
        tweet_word.append(word_count)
    return tweet_word



x_list = [] #list of tweets in string form from training group

for i in range(0,len(train)): 
    y = (train.iloc[i,3])
    x_list.append(str(y))



x_token = [] #list of list of token for each tweet

for x in x_list:
    x_token.append(twtToken(x))

word_list = make_word_list(x_token)


################################################################################
##make training sets
rows = []

for x in x_token:
    tweet_word = indexToken(x,word_list)
    rows.append(tweet_word)

x_train = pd.DataFrame(rows, columns = word_list) #data for naive bayes training


y_train = []
for y in train.sentiment:
    y_train.append(y)

###############################################################################
#make validation sets
v_list = []
for i in range(0, len(valid)):
    y = (valid.iloc[i,3])
    v_list.append(str(y))

    
v_token = []

for x in v_list:
    v_token.append(twtToken(x))

    
    
v_rows = []
for x in v_token:
    tweet_word = indexToken(x, word_list)
    v_rows.append(tweet_word)

x_test = pd.DataFrame(v_rows, columns = word_list)

y_test = []
for y in valid.sentiment:
    y_test.append(y)



#############################################################################
from joblib import dump

dump(x_train, "x_train.joblib")
dump(y_train, "y_train.joblib")
dump(x_test, "x_test.joblib")
dump(y_test, "y_test.joblib")
dump(word_list, "word_list.joblib")




























