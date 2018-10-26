import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

abstracts = []
types = []

# Training Data
with open('./dataset/train.dat', 'r') as f:
    for l in f:
        types.append(l[0])
        abstracts.append(l[2:])
dict_train = {'Type': types, 'Abstract': abstracts}
df_train = pd.DataFrame(dict_train)

abstracts = []
types = []

# Test Data
with open('./dataset/test.dat', 'r') as f:
    for l in f:
        abstracts.append(l[0:])
dict_test = {'Abstract': abstracts}
df_test = pd.DataFrame(dict_test)

print(df_train.head())
print(df_test.head())

stemmer = SnowballStemmer('english')
words = stopwords.words("english")

# Train
df_train['Abstract_Modified'] = df_train['Abstract'].apply(lambda x: " ".join(
    [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
X_train = df_train['Abstract_Modified']
y_train = df_train['Type']

# Test
df_test['Abstract_Modified'] = df_test['Abstract'].apply(lambda x: " ".join(
    [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
X_test = df_test['Abstract_Modified']


X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
    X_train, y_train, test_size=0.5, random_state=4)


X_train_train = pd.DataFrame(X_train_train)
X_train_test = pd.DataFrame(X_train_test)
y_train_train = pd.DataFrame(y_train_train)
y_train_test = pd.DataFrame(y_train_test)
X_test = pd.DataFrame(X_test)

print(X_train_train.head())
print(X_train_test.head())
print(X_train_train.shape)
print(y_train_train.shape)
print(y_train_test.shape)
print(X_test.shape)

from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

sgd_classifier = Pipeline([('vect', CountVectorizer(stop_words='english')),  ('tf_transformer', TfidfTransformer()),
                           ('clf', SGDClassifier())])
parameters = {'vect__max_df': (0.5, 0.75, 1.0),
              'vect__max_features': (None, 5000, 10000, 50000),
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1', 'l2'),
              'clf__penalty': ('l2', 'elasticnet'),
              }
sgd_classifier = sgd_classifier.fit(
    X_train_train['Abstract_Modified'], y_train_train)
predicted = sgd_classifier.predict(X_train_test['Abstract_Modified'])
f1_score(y_train_test, predicted, average='weighted')


# In[19]:


# Actual Prediction
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
sgd_classifier = Pipeline([('vect', CountVectorizer(stop_words='english')),  ('tf_transformer', TfidfTransformer()),
                           ('clf', SGDClassifier())])
parameters = {'vect__max_df': (0.5, 0.75, 1.0),
              'vect__max_features': (None, 5000, 10000, 50000),
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1', 'l2'),
              'clf__penalty': ('l2', 'elasticnet'),
              }
sgd_classifier = sgd_classifier.fit(
    df_train['Abstract_Modified'], df_train['Type'])
predictions = sgd_classifier.predict(df_test['Abstract_Modified'])

output_file = open('./predictions/predictions.dat', 'w')
for i in predictions:
    output_file.write(i + '\n')
output_file.close()

print(len(predictions))
