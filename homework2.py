#read in document
import pandas as pd

df = pd.read_csv("langid.csv")
docs = df['doc']
labels = df['language']

#TfIdfVectorizer default parameters
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb = MultinomialNB()
nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='word', ngram=(1,3), max_features=1000
tf = TfidfVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=1000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='word', ngram=(1,3), max_features=10000
tf = TfidfVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=10000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='word', ngram=(1,3), max_features=20000
tf = TfidfVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=20000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='char', ngram=(2,3), max_features=1000
tf = TfidfVectorizer(analyzer='char',
                    ngram_range=(2,3),
                    max_features=1000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='char', ngram=(2,3), max_features=10000
tf = TfidfVectorizer(analyzer='char',
                    ngram_range=(2,4),
                    max_features=10000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#TfIdfVectorizer analyzer='char', ngram=(2,3), max_features=20000
tf = TfidfVectorizer(analyzer='char',
                    ngram_range=(2,4),
                    max_features=20000)
X = tf.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer default parameters
from sklearn.feature_extraction.text import CountVectorizer

ct = CountVectorizer()
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb = MultinomialNB()
nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='word', ngram=(1,3), max_features=1000
ct = CountVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=1000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='word', ngram=(1,3), max_features=10000
ct = CountVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=10000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='word', ngram=(1,3), max_features=20000
ct = CountVectorizer(analyzer='word',
                    ngram_range=(1,3),
                    max_features=20000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='char', ngram=(2,3), max_features=1000
ct = CountVectorizer(analyzer='char',
                    ngram_range=(2,3),
                    max_features=1000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='char', ngram=(2,3), max_features=10000
ct = CountVectorizer(analyzer='char',
                    ngram_range=(2,3),
                    max_features=10000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#CountVectorizer analyzer='char', ngram=(2,3), max_features=20000
ct = CountVectorizer(analyzer='char',
                    ngram_range=(2,3),
                    max_features=20000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))

#Test best classifier (CountVectorizer analyzer='char', ngram=(2,3), max_features=10000)
ct = CountVectorizer(analyzer='char',
                    ngram_range=(2,3),
                    max_features=10000)
X = ct.fit_transform(docs)
X_train = X[:39102]
y_train = labels[:39102]
X_validate = X[39102:43991]
y_validate = labels[39102:43991]
X_test = X[43991:]
y_test = labels[43991:]

nb.fit(X_train,y_train)

y_test_pred = nb.predict(X_test)
print(classification_report(y_test,y_test_pred))
