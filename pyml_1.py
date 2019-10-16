# Import required libraries:
import json
import random
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Data:
# Create enum class:
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

# Create data class to neaten-up code:
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    # Define sentiment function:
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def even_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        neutral = list(filter(lambda x: x.sentiment == Sentiment.NEUTRAL, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))

        positive_shrunk = positive[:len(negative)]
        neutral_shrunk = neutral[:len(negative)]

        self.reviews = negative + positive_shrunk + neutral_shrunk
        random.shuffle(self.reviews)
        # print(len(negative))
        # print(len(positive_shrunk))
        # print(len(neutral_shrunk))

        # Check to see if working:
        # print(negative[0].text)
        # print(len(negative))
        # print(neutral[0].text)
        # print(len(neutral))
        # print(positive[0].text)
        # print(len(positive))

# Load Data:
# Save file to variable:
file_name = './data/sentiment/Books_small_10000.json'

# Create empty list for storing data:
reviews = []

# Open and read file:
with open(file_name) as f:
    # Read file line by line:
    for line in f:
        # Print first line to screen:
        # print("Line 1:")
        # print(line)
        # Print only reviewText to screen:
        review = json.loads(line)
        # print("Line 2:")
        # print(review['reviewText'])
        # print("Line 3:")
        # print(review['overall'])
        # Break out of loop to not print all reviews:
        # break
        # Add reviews and scores to list as an object:
        reviews.append(Review(review['reviewText'], review['overall']))

# To check if working:
# print(reviews[5].sentiment)

# Prep Data:
# Define and load initial test data:
training, testing = train_test_split(reviews, test_size = 0.33, random_state = 42)
train_cont = ReviewContainer(training)
train_cont.even_distribute()
test_cont = ReviewContainer(testing)
test_cont.even_distribute()
# print(len(cont.reviews))
# Get amount of train and test data used:
# print(len(training))
# print(len(testing))

# Pass training set into statements:
# But first test the above function:
# print(training[0].text)
# print(training[0].score)
# print(training[0].sentiment)

# Pass data for prediction using list-comprehention:
# train_x = [x.text for x in training]
train_x = train_cont.get_text()
# train_y = [x.sentiment for x in training]
train_y = train_cont.get_sentiment()

# Check above statement:
# print(train_x[0])
# print(train_y[0])
# print(train_y.count(Sentiment.NEGATIVE))
# print(train_y.count(Sentiment.NEUTRAL))
# print(train_y.count(Sentiment.POSITIVE))

# Pass data for prediction using list-comprehention:
# test_x = [x.text for x in testing]
test_x = train_cont.get_text()
# test_y = [x.sentiment for x in testing]
test_y = train_cont.get_sentiment()

# Check above statement:
# print(test_x[0])
# print(test_y[0])
# print(test_y.count(Sentiment.NEGATIVE))
# print(test_y.count(Sentiment.NEUTRAL))
# print(test_y.count(Sentiment.POSITIVE))

# Pass testing set into statements:
# But first test the above function:
# print(testing[0].text)
# print(testing[0].score)
# print(testing[0].sentiment)

# Bags-of-words vectorization:
# Initialize variable:
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

# Output defined training variable for useage:
# print(vectorizer.fit_transform(train_x))
train_x_vectors = vectorizer.fit_transform(train_x)
# print(train_x_vectors[0])
# print(train_x_vectors[0].toarray())
train_x_vectors_array = train_x_vectors.toarray()

# Output defined testing variable for useage:
# print(vectorizer.fit_transform(test_x))
test_x_vectors = vectorizer.transform(test_x)
# print(test_x_vectors[0])
# print(test_x_vectors[0].toarray())
test_x_vectors_array = test_x_vectors.toarray()

# Classification:
# Linear SVM:
# Define classifier:
clf_svm = svm.SVC(kernel='linear')
# Fit classifier to data:
clf_svm.fit(train_x_vectors, train_y)
# Prediciting classifier:
# print(train_x_vectors[0])
# print(clf_svm.predict(train_x_vectors[0]))

# Decision Tree:
# Define classifier:
clf_dec = DecisionTreeClassifier()
# Fit classifier to data:
clf_dec.fit(train_x_vectors, train_y)
# Prediciting classifier:
# print(train_x_vectors[0])
# print(clf_dec.predict(train_x_vectors[0]))

# Naive Bayes:
# Define classifier:
clf_gnb = GaussianNB()
# Fit classifier to data:
clf_gnb.fit(train_x_vectors_array, train_y)
# Prediciting classifier:
# print(train_x_vectors[0])
# print(clf_gnb.predict(train_x_vectors_array[0].reshape(1, -1)))

# Logistic Regression:
# Define classifier:
clf_lr = LogisticRegression()
# Fit classifier to data:
clf_lr.fit(train_x_vectors, train_y)
# Prediciting classifier:
# print(train_x_vectors[0])
# print(clf_lr.predict(train_x_vectors[0]))

# Evaluation:
# Mean accuracy:
# print(clf_svm.score(test_x_vectors, test_y))
# print(clf_dec.score(test_x_vectors, test_y))
# print(clf_gnb.score(test_x_vectors_array, test_y))
# print(clf_lr.score(test_x_vectors, test_y))
# F1 Score:
# print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]))
# print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]))
# print(f1_score(test_y, clf_gnb.predict(test_x_vectors_array), average=None, labels=[Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]))
# print(f1_score(test_y, clf_lr.predict(test_x_vectors), average=None, labels=[Sentiment.NEGATIVE, Sentiment.NEUTRAL, Sentiment.POSITIVE]))

# Checking ratios of sentiments:
# print(train_y.count(Sentiment.POSITIVE))
# print(train_y.count(Sentiment.NEUTRAL))
# print(train_y.count(Sentiment.NEGATIVE))

# Qualitative Analysis:
# operation_set = ["", "", ""]
# new_set = vectorizer.transform(operation_set)
# print(clf_svm.predict(new_set))

# Tuning the model with grid search:
parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 2, 4, 8, 16, 32)}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

# Saving Model:
with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('./models/sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

print(loaded_clf.predict(test_x_vectors[0]))
