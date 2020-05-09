##
# Author: Saatvik Tikoo
# Email: tikoo@usc.edu
##

# This is the main file that will train the models and give us the classifications.
# The data-set has been added but can be found at https://ai.stanford.edu/~amaas/data/sentiment/
# To run it follow these steps:
# 1. Install nltk and sklearn
# 2. After that we need to download few files from NlTK, so in python console do:
#       >> import nltk
#       >> nltk.download('stopwords')
#       >> nltk.download('punkt')
#       >> nltk.download('averaged_perceptron_tagger')
# 3. No simply run this file
# 4. It will take arounf 1 hour to generate the models and get the accuracies

from abc import ABC
from nltk.tag.stanford import StanfordPOSTagger
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import random
import pickle
import re
import threading

# Global variables
all_words = []
documents = []
# adjectives, adverbs and verbs are only allowed
allowed_word_types = ["J", "R", "V"]
# Getting Engilsh Language stemmer and the set of stop words
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


# Break the review set into chunks
def chunkIt(reviews):
    chunk_size = 1
    total_chunks = len(reviews) / float(chunk_size)
    out = []
    start = 0.0
    while start < len(reviews):
        out.append(reviews[int(start):int(start + total_chunks)])
        start += total_chunks
    return out


# define negative list
def negated_words(words):
    negate_list = []
    modifier = None
    negative_territory = 0
    total_words = len(words)
    for x in range(total_words):
        word = words[x]
        neg_verbs = ["n't", "not", "hardly"]
        if word in neg_verbs:
            modifier = "vrbAdj"
            negative_territory = 4
        neg_nouns = ["no", "none"]
        if word in neg_nouns:
            modifier = "nouns"
            negative_territory = 4
        if negative_territory > 0:
            pos = nltk.pos_tag([word])
            pos = pos[0][1]
            if (re.match('VB[G,P,D]*', pos) or re.match('JJ|RB', pos)) and modifier == "vrbAdj":
                if word not in stop_words:
                    negate_list.append(x)
            elif re.match('NN.*', pos) and modifier == "nouns":
                if word not in stop_words:
                    negate_list.append(x)
            negative_territory -= 1
    return negate_list


# Parse positive reviews
def parse_pos_reviews(pos_review_chunk):
    global documents
    for p in pos_review_chunk:
        documents.append((p, 'pos'))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        negate_list = set(negated_words(words))
        j = 0
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words:
                    if j in negate_list:
                        all_words.append("not_" + (stemmer.stem(w[0].lower())))
                    else:
                        all_words.append(stemmer.stem(w[0].lower()))
            j += 1


# Parse negative reviews
def parse_neg_reviews(neg_review_chunk):
    global documents
    for p in neg_review_chunk:
        documents.append((p, 'neg'))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        negate_list = set(negated_words(words))
        j = 0
        for w in pos:
            if w[1][0] in allowed_word_types:
                if w[0] not in stop_words:
                    if j in negate_list:
                        all_words.append("not_" + (stemmer.stem(w[0].lower())))
                    else:
                        all_words.append(stemmer.stem(w[0].lower()))
            j += 1


# Define multithreading while parsing the reviews
def multiThread_parse(pos_r, neg_r):
    # Positive Comments
    pos_review_group = chunkIt(pos_r)
    length_pos_review_group = len(pos_review_group)
    threads = []
    for i in range(length_pos_review_group):
        threads.append(threading.Thread(target=parse_pos_reviews, args=(pos_review_group.pop(),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Negative Comments
    neg_review_group = chunkIt(neg_r)
    length_neg_review_group = len(neg_review_group)
    threads = []
    for i in range(length_neg_review_group):
        threads.append(threading.Thread(target=parse_neg_reviews, args=(neg_review_group.pop(),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# Find the features in dataset
def find_features(document):
    words = word_tokenize(document)
    words_set = set()
    negate_list = set(negated_words(words))
    j = 0
    for word in words:
        if word not in stop_words:
            if j in negate_list:
                words_set.add("not_" + (stemmer.stem(word.lower())))
            else:
                words_set.add(stemmer.stem(word.lower()))
        j += 1
    features = {}
    for w in word_features:
        features[w] = (w in words_set)
    return features


class VoteClassifier(ClassifierI, ABC):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


if __name__ == '__main__':
    path_to_model = 'stanford-postagger/models/english-bidirectional-distsim.tagger'
    path_to_jar = 'stanford-postagger/stanford-postagger.jar'
    tagger = StanfordPOSTagger(path_to_model, path_to_jar)
    tagger.java_options = '-mx512m'

    # Load the data-set
    pos_review = open('short_reviews/train/positive.txt', encoding='ISO-8859-1').readlines()
    neg_review = open('short_reviews/train/negative.txt', encoding='ISO-8859-1').readlines()

    multiThread_parse(pos_review, neg_review)
    # Save all the adjectives to a file
    with open("trained_models/documents.pickle", "wb") as save_documents:
        pickle.dump(documents, save_documents)

    all_words = nltk.FreqDist(all_words)
    print(len(list(all_words.keys())))
    word_features = list(all_words.keys())[:6000]

    save_word_features = open("trained_models/word_features6k.pickle", "wb")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()
    print("saved word features!")

    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    random.shuffle(featuresets)
    print(len(featuresets))
    testing_set = featuresets[10000:]
    training_set = featuresets[:10000]

    # Logistic Regression
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:",
          (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

    save_classifier = open("trained_models/LogisticRegression_classifier5k.pickle", "wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()

    # Linear SVC
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

    save_classifier = open("trained_models/LinearSVC_classifier5k.pickle", "wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()

    # Kernal SVC
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

    save_classifier = open("trained_models/NuSVC_classifier5k.pickle", "wb")
    pickle.dump(SVC_classifier, save_classifier)
    save_classifier.close()

    # SGD
    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(training_set)
    print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

    save_classifier = open("trained_models/SGDC_classifier5k.pickle", "wb")
    pickle.dump(SGDC_classifier, save_classifier)
    save_classifier.close()

    # Random Forest
    RF_classifier = SklearnClassifier(RandomForestClassifier())
    RF_classifier.train(training_set)
    print("RFClassifier accuracy percent:", nltk.classify.accuracy(RF_classifier, testing_set) * 100)

    save_classifier = open("trained_models/RF_classifier5k.pickle", "wb")
    pickle.dump(RF_classifier, save_classifier)
    save_classifier.close()

    voted_classifier = VoteClassifier(SVC_classifier, LinearSVC_classifier,
                                      SGDC_classifier, LogisticRegression_classifier, RF_classifier,)

    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

    print("Classification for the first few rows from Data set are: ")
    print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0]) * 100)
    print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0]) * 100)
    print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0]) * 100)
    print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0]) * 100)
    print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0]) * 100)
    print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0]) * 100)
