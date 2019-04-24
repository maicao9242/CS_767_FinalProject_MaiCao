#import modules
from twython import Twython
import json
import pandas as pd
import os

#set path to save result into a csv file
path= r'C:\Users\kemly\Desktop\CS767\FINAL PROJECT'

#open JSON file containing user authentication key information
with open("token.json","r") as file:
    creds = json.load(file)

#use authentication to get access token in order to access Twitter API
twitter = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'], oauth_version = 2)
ACCESS_TOKEN = twitter.obtain_access_token()
twitter = Twython(creds['CONSUMER_KEY'], access_token = ACCESS_TOKEN)


#query to search for a keyword
query = {'q': 'woman',  #keyword
        'result_type': 'mixed', #get both popular results and recent results to make sure the likes and retweets vary between tweets
        'count': 100,
        'lang': 'en',
        }

#query Twitter for the keyword and save the username,tweet text, number of likes, number of retweets into a dict
dict_ = {'user': [], 'text': [], 'favorite_count': [], 'retweet_count':[]}
for status in twitter.search(**query)['statuses']:  
    dict_['user'].append(status['user']['screen_name'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])
    dict_['retweet_count'].append(status['retweet_count'])

# create a pandas DataFrame of the result and save it into a csv file
df = pd.DataFrame(dict_) #create a pandas dataframe using the 

df.to_csv(os.path.join(path, r'man.csv')) #save file with keyword as name


for d in range(0,len(dict_['user'])):
    print(dict_['user'][d] + ": " + dict_['text'][d])
    print("Favorite count: " + str(dict_['favorite_count'][d]))
    print("Retweet count: " + str(dict_['retweet_count'][d]))
    print("_______________________________")




