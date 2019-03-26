#read in file
import pandas as pd

df = pd.read_csv("langid.csv")

#split file in train-validate-test set
docs = df['doc']
docs_train = docs[:39102]
docs_validate = docs[39102:43991]
docs_test = docs[43991:]

labels = df['language']
labels_train = labels[:39102]
labels_validate = labels[39102:43991]
labels_test = labels[43991:]

#vectorize file
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_train = tf.fit_transform(docs_train)
X_validate = tf.fit_transform(docs_validate)
X_test = tf.fit_transform(docs_test)
y_train = tf.fit_transform(labels_train)
y_validate = tf.fit_transform(labels_validate)
y_test = tf.fit_transform(labels_test)

#train classifier and validate (doesn't work)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb = MultinomialNB()
nb.fit(X_train,y_train)

y_validate_pred = nb.predict(X_validate)
print(classification_report(y_validate,y_validate_pred))