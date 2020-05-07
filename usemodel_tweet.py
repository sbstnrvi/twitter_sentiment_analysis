# Import library yang dibutuhkan
import tweepy
import re
import pickle
from tweepy import OAuthHandler
from sklearn.datasets import load_files
import preprocessor as p
import numpy as np

# Mohon DIGANTI dengan consumer key, consumer secret, access token, dan access secret milik kalian sendiri ya
consumer_key = 'NXKFSY2HhvR9ToWXIgoU2dg0c'
consumer_secret = 'JosjRuDqrMUBCErYIOBC5vL3jl2tFPZSWife3u36UFabAYdwAV' 
access_token = '100127632-gUZP9zM5iCxZanmeO45XyhXwSyGAhNiCVp3vekKK'
access_secret ='lSauyi7XVER0Zn4PO7vYd4HD1mgcVTp7u3jA61VEXzZ6n'
  
# Initializing the tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
args = ['pemerintah corona'];
api = tweepy.API(auth,timeout=10)

# Buat list kosong untuk menampung tweets
list_tweets = []

# Menarik data tweet dari twitter
query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='id',result_type='recent',tweet_mode='extended').items(10):
        list_tweets.append(status.full_text)

#tweet preprocessing (membersihkan tweet)
tweets_preprocessed=[]
for tweets in list_tweets:
    tweets=p.clean(tweets)
    tweets_preprocessed.append(tweets)

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()      

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factorySW = StopWordRemoverFactory()
stopword = factorySW.create_stop_word_remover()

corpus = []
unicode=[u"\u0131",u"\u015f"]        
for tweet in tweets_preprocessed:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = stemmer.stem(tweet)
    tweet = stopword.remove(tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    for h in unicode:
        tweet = tweet.replace(h, "")    
    corpus.append(tweet)
    
# Loading the vectorizer and classfier
with open('classifier_knn.pickle','rb') as f:
    classifier_knn = pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)  

X=tfidf.transform(corpus).toarray()    
sent_predict=classifier_knn.predict(X)

def myfunc(a,b):
    if a<b:
        return "Negatif"
    else:
        return "Positif"
    
vfunc=np.vectorize(myfunc)
sent_predict_text=vfunc(sent_predict, 1)    

for j in range(len(list_tweets)):
    print("tweet : {}".format(list_tweets[j]))
    print("sentimen : {} \n".format(sent_predict_text[j]))