
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split


# In[2]:


mylist2=[]
mylist1=[]

#Train
with open('/Users/uttara/desktop/train.dat','r') as f:
    for l in f:
        mylist1.append(l[0])
        mylist2.append(l[2:])
d1={'Type':mylist1,'Abstract':mylist2}
df_train=pd.DataFrame(d1)

mylist2=[]
mylist1=[]
#Test
with open('/Users/uttara/desktop/test.dat','r') as f:
    for l in f:
        mylist2.append(l[0:])
d2={'Test_Abstract':mylist2}
df_test=pd.DataFrame(d2)


# In[3]:


df_train.head()


# In[4]:


df_test.head()


# In[5]:


stemmer = SnowballStemmer('english')
words = stopwords.words("english")

#Train
# df_train['Abstract_New']=df_train['Abstract']
df_train['Abstract_New'] = df_train['Abstract'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
X1=df_train['Abstract_New']
y1=df_train['Type']


# In[6]:


#Test
# df_test['Abstract_Test_New']=df_test['Test_Abstract']
df_test['Abstract_Test_New'] = df_test['Test_Abstract'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
X2=df_test['Abstract_Test_New']


# In[7]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.5,random_state=4)


# In[8]:


X1_train=pd.DataFrame(X1_train)
X1_test=pd.DataFrame(X1_test)
y1_train=pd.DataFrame(y1_train)
y1_test=pd.DataFrame(y1_test)
X2=pd.DataFrame(X2)


# In[9]:


df_train.shape


# In[10]:


df_test.shape


# In[11]:


X1_train.head()


# In[12]:


X1_test.head()


# In[13]:


X1_train.shape


# In[14]:


y1_train.shape


# In[15]:


y1_test.shape


# In[16]:


X2.shape


# In[18]:


from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
myclf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),  ('tf_transformer', TfidfTransformer()), 
                    ('clf', SGDClassifier())])
parameters={ 'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__penalty': ('l2', 'elasticnet'),
           }
myclf_svm = myclf_svm.fit(X1_train['Abstract_New'], y1_train)
predicted = myclf_svm.predict(X1_test['Abstract_New'])
f1_score(y1_test,predicted,average='weighted')


# In[19]:


#Actual Prediction
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
myclf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),  ('tf_transformer', TfidfTransformer()), 
                    ('clf', SGDClassifier())])
parameters={ 'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__penalty': ('l2', 'elasticnet'),
           }
myclf_svm = myclf_svm.fit(df_train['Abstract_New'], df_train['Type'])
dfTest_predicted = myclf_svm.predict(df_test['Abstract_Test_New'])


# In[20]:


myfile = open ('/Users/uttara/desktop/ActualFormat.dat', 'w')
for i in dfTest_predicted:
    myfile.write(i +'\n')
myfile.close()


# In[21]:


len(dfTest_predicted)

