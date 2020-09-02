# coding: utf-8

import sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline 


# Categories, change these to match the folder names you have in the categories folder.
categories = ['forgiveness', 'geneology', 'love-of-god', 'prayer', 'salvation', 'second-coming', 'self-control']

docs_to_train = sklearn.datasets.load_files("categories", 
    description=None, categories=categories, 
    load_content=True, encoding='utf-8', shuffle=True, random_state=42)

# Train our data, and set the percentage of  data that should be used to test our training.
X_train, X_test, y_train, y_test = train_test_split(docs_to_train.data,
    docs_to_train.target, test_size=0.4)


count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(raw_documents=X_train)


tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

count_vect = CountVectorizer(stop_words='english')
X_test_counts = count_vect.fit_transform(raw_documents=X_test)

tfidf_transformer = TfidfTransformer(use_idf=True)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

# The classifier to use, modifying classifiers and their parameters will can improve accuracy.
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, 
    verbose=0)),])

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

# Print stats
print ('Unoptimized Score: ' + str(np.mean(predicted == y_test)))

# Metrics for each category
# print(metrics.classification_report(y_test, predicted, 
#      target_names=docs_to_train.target_names))

# Import and use Grid Search to optimize params for your classifier.
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
  'tfidf__use_idf': (True, False),
  'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(docs_to_train.data, docs_to_train.target)

# The most optimized score
print('Optimized Score: ' + str(gs_clf.best_score_))

# The best params to use in optimization for our classifier
# print(gs_clf.best_params_)


# Test our classification based on new input.
input = [
  'for God so loved the world that',
  'james the son of alpheus',
  'pray without ceasing']

predicted = gs_clf.predict(input);
output = [ categories[i] for i in predicted]

print ('New Input: ' + str(input))
print ('Category Prediction: ' + str(output));
