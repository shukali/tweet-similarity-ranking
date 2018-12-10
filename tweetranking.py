import numpy as np
from numpy.linalg import norm
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

n_tweets_to_read = 5000 # read the first 5000 tweets

cosine_similarity = lambda a, b: np.inner(a, b) / norm(a) * norm(b) if norm(a) != 0.0 and norm(b) != 0.0 else 0.0

def TermDocumentMatrix(docs, docIDs=None):
    '''
    Creates a term-document matrix of size (n_documents, |Vocabulary|) as a pandas 
    dataframe. The columns will correspond to the terms and the rows will be 
    accessible as the given document IDs, if given. The entry (d, t) counts the 
    number of occurences of t in document d.
    '''
    vectorizer = CountVectorizer(lowercase=True, stop_words=None)
    tdm = vectorizer.fit_transform(docs)
    tdm_feature_names = vectorizer.get_feature_names()
    #
    df = pd.DataFrame(tdm.toarray(), columns=tdm_feature_names, dtype="float64")
    if docIDs is not None:
        df.index = docIDs    
    return df

#init
ps = PorterStemmer()

# read input file linewise and store in list
tweets = []
with open("data/tweets.csv", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i < n_tweets_to_read:
            tweets.append(line)
        else:
            break  
tweets = list(set(tweets)) # remove duplicate tweets
print("{} unique Tweets loaded\n".format(len(tweets)))

# preprocessing
tweetIDs = []
tweetsProcessed = []
for tweet in tweets:
    doc = tweet.split("\t")   # split by tab
    tweetIDs.append(doc[1])     # add ID to list
    doc = doc[3:]   # remove date, time, +utc, tweetID, author
    tok_doc = word_tokenize(" ".join(doc))    # tokenize remaining document
    stemmed_doc = [ps.stem(word) for word in tok_doc] 
    tweetsProcessed.append(" ".join(stemmed_doc))   # set tweet to be stemmed words

# term frequency
tdf = TermDocumentMatrix(tweetsProcessed, tweetIDs)

# calculate document frequency
documentFrequencies = []
for index, series in tdf.iteritems(): 
    # store df value, number of non-zero values
    documentFrequencies.append(len(series.nonzero()[0])) 

# calculate tf-ids weight
tdf.applymap(lambda x: 1.0 + np.log10(x) if x > 0.0 else 0.0) # log frequency weight
idf = pd.Series(np.log10(len(tweets)/np.array(documentFrequencies))) # inverse document frequency
tf_idf = tdf * idf.values
#tf_idf = tf_idf.div(tf_idf.sum(axis='columns'), axis='index') # normalize matrix

def CosineSimilarityOfTweets(tweet1=tweets[0], tweet2=tweets[1]):
    '''
    Calculates the similarity of the two tweets based on the cosine similarity 
    and the given tf-idf matrix (n_documents, |V|).
    '''
    ID1 = tweetIDs[tweets.index(tweet1)]
    ID2 = tweetIDs[tweets.index(tweet2)]
    return cosine_similarity(tf_idf.loc[[ID1]], tf_idf.loc[[ID2]])

#print(CosineSimilarityOfTweets(tweets[0], tweets[0])) # identical
#print(CosineSimilarityOfTweets(tweets[49], tweets[50])) # very similar
#print(CosineSimilarityOfTweets(tweets[71], tweets[1])) # quite different

def printTopSimilarTweets(tweetID='965706998946893824', n=10):
    '''
    Prints the top n tweets from the tweets dataset that are most similar to the 
    given ID's tweet. The result is shown decreasing by the cosine similarity 
    of the tf-idf values.
    '''
    result = tf_idf.apply(lambda row: cosine_similarity(tf_idf.loc[[tweetID]], row), 
                          axis='columns').sort_values(ascending=False)
    print("Query: " + tweets[tweetIDs.index(tweetID)] + "\n")
    for i in range(0, n):
        print("{}: ".format(i+1) + tweets[tweetIDs.index(result.index[i])] + "\n")

# output
printTopSimilarTweets(tweetID='965734505205063680')
while True:
    id = input("Please enter the tweet ID to perform similarity search for:")
    if id in tf_idf.index:
        printTopSimilarTweets(tweetID=id)
    print("\n\n")