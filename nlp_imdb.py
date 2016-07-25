# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:54:04 2016

@author: rghiglia
"""

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd

dnm = r'C:\Users\rghiglia\Documents\ML_ND\Kaggle\NLP'
fnm = r'labeledTrainData.tsv'
fnmL = dnm + '\\' + fnm
train = pd.read_csv(fnmL, header=0, delimiter="\t", quoting=3)

train.info()

train['review'][0]

# ******************************************
# Data Cleaning and Text Preprocessing
# ******************************************

# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup


# -------------------------------------------------
# HTML
# -------------------------------------------------

# Initialize the BeautifulSoup object on a single movie review     
example1 = BeautifulSoup(train['review'][0])

# Print the raw review and then the output of get_text(), for 
# comparison
print train["review"][0]
print example1.get_text()
# Ok, cleaned up of HTML stuff

# -------------------------------------------------
# Dealing with Punctuation, Numbers and Stopwords: NLTK and regular expressions
# -------------------------------------------------

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub('[^a-zA-Z]',           # The pattern to search for
                      ' ',                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print letters_only


#We'll also convert our reviews to lower case and split them into individual words (called "tokenization" in NLP lingo):

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()               # Split into words



# -------------------------------------------------
# Stop words (words with little particular meaning)
# -------------------------------------------------

#Finally, we need to decide how to deal with frequently occurring words that don't carry much meaning. Such words are called "stop words"; in English they include words such as "a", "and", "is", and "the". Conveniently, there are Python packages that come with stop word lists built in. Let's import a stop word list from the Python Natural Language Toolkit (NLTK). You'll need to install the library if you don't already have it on your computer; you'll also need to install the data packages that come with it, as follows:
import nltk
#nltk.download()
from nltk.corpus import stopwords # Import the stop word list
print stopwords.words("english")

# Remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print words


# -------------------------------------------------
# Stemming
# -------------------------------------------------
# Review this
#There are many other things we could do to the data - For example, Porter Stemming and Lemmatizing (both available in NLTK) would allow us to treat "messages", "message", and "messaging" as the same word, which could certainly be useful. However, for simplicity, the tutorial will stop here.




stops = set(stopwords.words("english"))
def review_to_words(raw_review, stops=set(stopwords.words("english"))):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # # Put it in input, sO it doesn't do it every time
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( ' '.join( meaningful_words ))   


clean_review = review_to_words( train['review'][0] )
print clean_review


# Get the number of reviews based on the dataframe column size
nR = len(train)

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range(nR):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    print 'Review {}/{}'.format(i, nR)
    clean_train_reviews.append( review_to_words( train["review"][i] ) )


# -------------------------------------------------
# Converting to numeric representation
# -------------------------------------------------

#Creating Features from a Bag of Words (Using scikit-learn)
#
#Now that we have our training reviews tidied up, how do we convert them to some kind of numeric representation for machine learning? One common approach is called a Bag of Words. The Bag of Words model learns a vocabulary from all of the documents, then models each document by counting the number of times each word appears. For example, consider the following two sentences:
#
#Sentence 1: "The cat sat on the hat"
#
#Sentence 2: "The dog ate the cat and the hat"
#
#From these two sentences, our vocabulary is as follows:
#
#{ the, cat, sat, on, hat, dog, ate, and }
#
#To get our bags of words, we count the number of times each word occurs in each sentence. In Sentence 1, "the" appears twice, and "cat", "sat", "on", and "hat" each appear once, so the feature vector for Sentence 1 is:
#
#{ the, cat, sat, on, hat, dog, ate, and }
#
#Sentence 1: { 2, 1, 1, 1, 1, 0, 0, 0 }
#
#Similarly, the features for Sentence 2 are: { 3, 1, 0, 0, 1, 1, 1, 1}

print 'Creating the bag of words...\n'
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
#type(train_data_features) # scipy.sparse.csr.csr_matrix

train_data_features = train_data_features.toarray()
#type(train_data_features) # numpy.ndarray

print train_data_features.shape

# You can check the vocabulary with:
#vectorizer.vocabulary_
# Maybe not what I think it is ... check documentation

# Or:
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)
# Yeah, although you need to combine it with the vocabulary to make it intelligible
# Are they in the same order?

voc_cnt = {vocab[i]: dist[i] for i in range(len(vocab))}

# Check:
vectorizer.vocabulary_['deal']
# NOPE!!

# His version:
# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

#vectorizer.vocabulary_['zone']


print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)


# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )


# Read the test data
fnm = r'testData.tsv'
fnmL = dnm + '\\' + fnm
test = pd.read_csv(fnmL, header=0, delimiter="\t", quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test)
clean_test_reviews = [] 


print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

