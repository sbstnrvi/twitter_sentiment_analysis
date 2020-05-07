# Import library yang dibutuhkan
import tweepy
import re
import pickle
from tweepy import OAuthHandler
from sklearn.datasets import load_files
import preprocessor as p

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
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='id',result_type='recent',tweet_mode='extended').items(200):
        list_tweets.append(status.full_text)

#tweet preprocessing (membersihkan tweet)
tweets_preprocessed=[]
for tweets in list_tweets:
    tweets=p.clean(tweets)
    tweets_preprocessed.append(tweets)

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
    tweet = re.sub(r"\s+"," ",tweet)
    for h in unicode:
        tweet = tweet.replace(h, "")    
    corpus.append(tweet)
    
#Kita simpan data tweets yang sudah kita dapatkan ke dalam penyimpanan local kita
for i in range(len(corpus)):
    with open('tweet/tweets_{}.txt'.format(i),'w') as f:
        f.write(corpus[i])
     

# Import dataset dari tweet yang sudah kita klasifikasikan tadi
reviews = load_files('tweet/')
X,y = reviews.data,reviews.target    

# Pickling the dataset
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)

# Unpickling dataset
X_in = open('X.pickle','rb')
y_in = open('y.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)


# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()      

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factorySW = StopWordRemoverFactory()
stopword = factorySW.create_stop_word_remover()
 
corpus2=[]
for i in range(len(X)):
    tweetx = re.sub(r'\W', ' ', str(X[i]))
    tweetx = re.sub(r'^br$', ' ', tweetx)
    tweetx = re.sub(r'\s+br\s+',' ',tweetx)
    tweetx = re.sub(r'\s+[a-z]\s+', ' ',tweetx)
    tweetx = re.sub(r'^b\s+', '', tweetx)
    tweetx = re.sub(r'\s+', ' ', tweetx)         
    tweetx = stemmer.stem(tweetx)
    tweetx = stopword.remove(tweetx)
    corpus2.append(tweetx)
 
# Membuat model Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6)
X = vectorizer.fit_transform(corpus2).toarray()
        
# membagi data train dan data test
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

""" REGRESI LOGISTIK """
from sklearn.linear_model import LogisticRegression
classifier_logreg = LogisticRegression(random_state = 0)
classifier_logreg.fit(text_train, sent_train)
 
# Memprediksi hasil modelnya ke test set
y_pred_logreg = classifier_logreg.predict(text_test)
 
# Membuat Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm_logreg = confusion_matrix(sent_test, y_pred_logreg)
as_logreg = accuracy_score(sent_test, y_pred_logreg)

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_knn.fit(text_train, sent_train)
 
# Memprediksi Test set
y_pred_knn = classifier_knn.predict(text_test)
 
# Membuat Confusion Matrix
cm_knn = confusion_matrix(sent_test, y_pred_knn)
as_knn = accuracy_score(sent_test, y_pred_knn)

""" SVM GAUSSIAN """
from sklearn.svm import SVC
classifier_svm_gaussian = SVC(kernel = 'rbf', random_state = 0)
classifier_svm_gaussian.fit(text_train, sent_train)
 
# Memprediksi hasil test set
y_pred_svm_gaussian = classifier_svm_gaussian.predict(text_test)
 
# Membuat confusion matrix
cm_svm_gaussian = confusion_matrix(sent_test, y_pred_svm_gaussian)
as_svm_gaussian = accuracy_score(sent_test, y_pred_svm_gaussian)

""" SVM NO KERNEL """
from sklearn.svm import SVC
classifier_svm_nokernel = SVC(kernel = 'linear', random_state = 0)
classifier_svm_nokernel.fit(text_train, sent_train)
 
# Memprediksi hasil test set
y_pred_svm_nokernel = classifier_svm_nokernel.predict(text_test)
 
# Membuat confusion matrix
cm_svm_nokernel = confusion_matrix(sent_test, y_pred_svm_nokernel)
as_svm_nokernel = accuracy_score(sent_test, y_pred_svm_nokernel)

""" NAIVE BAYES """
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(text_train, sent_train)
 
# Memprediksi hasil test set
y_pred_nb = classifier_nb.predict(text_test)
 
# Membuat confusion matrix
cm_nb = confusion_matrix(sent_test, y_pred_nb)
as_nb = accuracy_score(sent_test, y_pred_nb)

""" DECISION TREE CLASSIFICATION """
from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dtc.fit(text_train, sent_train)
 
# Memprediksi hasil test set
y_pred_dtc = classifier_dtc.predict(text_test)
 
# Membuat confusion matrix
cm_dtc = confusion_matrix(sent_test, y_pred_dtc)
as_dtc = accuracy_score(sent_test, y_pred_dtc)

""" RANDOM FOREST CLASSIFIER """
from sklearn.ensemble import RandomForestClassifier
classifier_rfc = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier_rfc.fit(text_train, sent_train)
 
# Memprediksi hasil test set
y_pred_rfc = classifier_rfc.predict(text_test)
 
# Membuat confusion matrix
cm_rfc = confusion_matrix(sent_test, y_pred_rfc)
as_rfc = accuracy_score(sent_test, y_pred_rfc)


score={'as_logreg':as_logreg, 'as_knn':as_knn, 'as_svm_gaussian':as_svm_gaussian, 'as_svm_nokernel':as_svm_nokernel, 'as_nb':as_nb, 'as_dtc':as_dtc, 'as_rfc':as_rfc}
score_list=[]
for i in score:
    score_list.append(score[i])
    u=max(score_list)
    if score[i]==u:
        v=i  
    print(f"{i}={score[i]}");   
print(f"The best method to use in this case is {v} with accuracy score {u}")


# Menyimpan classifier knn
with open('classifier_knn.pickle','wb') as f:
    pickle.dump(classifier_knn,f)
    
# Menyimpan model Tf-Idf
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)  

