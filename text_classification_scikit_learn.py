#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:53:36 2021

@author: selin
"""


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Firstly, clean the data and take only required columns for Text classification
# Sorted data for this research is avaliable in GitHub working folder and can be used as example
path_of_xlsx_data = ' ' #add your working directory here
df = pd.read_excel(path_of_xlsx_data)
df = df.drop(columns=['username','id','permalink','date'])
df = df.dropna()
df = df[pd.notnull(df['toxicity'])]
df['category'] = df.toxicity.map({1: 'social', 2: 'STEM', 3: 'unknown'})
df['category']=df['category'].astype('str')
df['toxicity']=df['toxicity'].astype('int')
category_id_df = df[['category', 'toxicity']].drop_duplicates().sort_values('toxicity')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['toxicity', 'category']].values)

# Then check the data distribution between classes, in case of inbalanced classes.
# If class contain not enoth data, the classification will be less accurate.

fig = plt.figure(figsize=(8,6))
df.groupby('toxicity').text.count().plot.bar(ylim=0)
plt.show()


# Creation of Term Frequency, Inverse Document Frequency (tf-idf)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.text).toarray()
labels = df.toxicity


# Model Selection - which model is better for this case.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

# Here we can numerically compare accuracy percentage and choose the most precise model

cv_df.groupby('model_name').accuracy.mean()

# In my case i choose the (Multinomial) Naive Bayes model
# Here we create confusion matrix which shows differecnce between predicted and actual labels.
# Mostly predictions are on a diagonal, where predicted label = actual label. 
# More accurace can be reached by adding more data of poor classes or deleting controversial data which represents both classes at once.

model = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification report for each class

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['category'].unique()))


#Source: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

