import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt

import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline,linear_model
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Organize all articles
train_articles = os.listdir("datasets/train-articles")
train_labels_tags_span = os.listdir("datasets/train-labels-task1-span-identification")
train_tags_technique = os.listdir("datasets/train-labels-task2-technique-classification")
train_articles.sort()
train_labels_tags_span.sort()

# Create dictionary of all articles with the key being their numbers.
propTagsSpan = {}

for i in range(len(train_articles)):
    article = train_articles[i]

    # removing .txt file extension
    articleNoExt = os.path.splitext(article)[0]

    # replace train articles with the same name
    train_articles[i] = articleNoExt

    # removing article
    articleNo = articleNoExt.replace('article', '')
    tagPath = "datasets/train-labels-task1-span-identification/" + articleNoExt + ".task1-SI.labels"

    with open(tagPath) as r:
        tags = r.readlines()
        for i in range(len(tags)):
            tag = tags[i]
            tag = tag.replace("\t", " ")
            tag = tag.replace("\n", " ")
            tags[i] = tag
        propTagsSpan[articleNoExt] = tags
    r.close()

propagandaTagTechnique = os.listdir("datasets/train-labels-task2-technique-classification")
propagandaTagTechnique.sort()
propTagsTechnique = {}

for i in range(len(train_articles)):
    article = train_articles[i]
    # removing .txt file extension
    articleNoExt = os.path.splitext(article)[0]
    # replace train articles with the same name
    train_articles[i] = articleNoExt
    # removing article
    articleNo = articleNoExt.replace('article', '')
    tagPath = "datasets/train-labels-task2-technique-classification/" + articleNoExt + ".task2-TC.labels"

    with open(tagPath) as r:
        tags = r.readlines()
        for i in range(len(tags)):
            tag = tags[i]
            tag = tag.replace(articleNo, " ")
            tag = tag.replace("\t", " ")
            tag = tag.replace("\n", " ")
            tags[i] = tag
        propTagsTechnique[articleNoExt] = tags
    r.close()

propaganda_sent_span = []

for article in train_articles:
    article_path = "datasets/train-articles/" + article + ".txt"
    tags = propTagsSpan[article]

    with open(article_path, encoding="utf-8") as r:
        entireArticle = r.read()
        for tag in tags:
            tag = tag.split()
            start = int(tag[1])
            end = int(tag[2])

            tag_line = entireArticle[start:end]
            tag_line = tag_line.replace("\n", " ")
            tag_line = tag_line.replace("\t", " ")

            propaganda_sent_span.append(tag_line)
    r.close()

propoganda_techniques = {}
propoganda_techniques["Sentence"] = []
propoganda_techniques["Technique"] = []

for article in train_articles:
    article_path = "datasets/train-articles/" + article + ".txt"
    tags = propTagsTechnique[article]

    with open(article_path, encoding="utf-8") as r:
        entireArticle = r.read()
        for tag in tags:
            tag = tag.split()
            propoganda_techniques["Technique"].append(tag[0])
            start = int(tag[1])
            end = int(tag[2])

            tag_line = entireArticle[start:end]
            tag_line = tag_line.replace("\n", " ")
            tag_line = tag_line.replace("\t", " ")
            propoganda_techniques["Sentence"].append(tag_line)
    r.close()

df = pd.DataFrame.from_dict(propoganda_techniques)
print(df)

propoganda_techniques_tags = ['Appeal_to_Authority','Name_Calling,Labeling','Slogans', 'Loaded_Language','Appeal_to_fear-prejudice','Repetition','Doubt','Exaggeration,Minimisation','Flag-Waving','Causal_Oversimplification','Whataboutism,Straw_Men,Red_Herring','Black-and-White_Fallacy','Thought-terminating_Cliches','Bandwagon,Reductio_ad_hitlerum']
plt.figure(figsize=(10,4))
df.Technique.value_counts().plot(kind='bar');

nltk.download("stopwords")
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if
                    word not in STOPWORDS)  # delete stopwors from text - not sure if we want to do this
    return text


df['Sentence'] = df['Sentence'].apply(clean_text)
df['Sentence'].apply(lambda x: len(x.split(' '))).sum()

X = df.Sentence
y = df.Technique
#random_state sets a seed, the train-test splits are always deterministic. If the seed is not set, train-test splits are different each time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=propoganda_techniques_tags))
print(f1_score(y_test, y_pred, average='weighted'))

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='lbfgs', multi_class='auto')),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=propoganda_techniques_tags))
print(f1_score(y_test, y_pred, average='weighted'))