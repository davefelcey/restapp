import urllib.request
import html2text
import re
import pandas as pd
import numpy as np
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import language_check

# Get job details

url = 'https://www.indeed.co.uk/viewjob?jk=b39ccb985a9de5e1&from=tp-serp&tk=1c29ob0im14he6rl'
job = ''

with urllib.request.urlopen(url) as resource:
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_links = True
    h.ignore_emphasis = True
    h.ignore_anchors = True
    h.drop_white_space = True
    h.unicode_snob = True

    content = resource.read()
    charset = resource.headers.get_content_charset()
    content = content.decode(charset)
    text = h.handle(content)
    job = text

# Format job

def formatSentence(t):
    return "\"{0}\",\"FALSE\"\n".format(t)

def sentenceTokenizer(t):
    result = ""
    # Clean no alpha numeric and punctuation 
    text = re.sub(r"[^A-Za-z0-9.?!:\n()-/ *]+", " ", t)
    # Split sentences
    sentenceList = re.split(r"[!?:.*]", text)
    for s in sentenceList:
        # Strip out excess white space
        sentence = s.strip()
        # Ignore linces where no text
        words = re.findall(r'[A-Za-z]+', sentence)
        if not words:
            continue
        # Split any multiple sentence on blank line boundary
        subSentList = re.split(r"\n\s*\n|\r\n\s\r\n", sentence)
        if subSentList:
            for subSent in subSentList:		
                subSent = subSent.strip()
                if subSent:
                    text = re.sub(r"[-/\n\r]+", " ", subSent)
                    result = result + formatSentence(text)
                elif sentence:
                    text = re.sub(r"[\n\r]+", " ", sentence)
                    result = result + formatSentence(text)
    return result

jobCSVData = sentenceTokenizer(job)

# Create model

trainFilepath = 'train-jobs.csv'

# Load and randomize trainng data
train = pd.read_csv(trainFilepath, header=None, names=['Text','Requirement'])
train = train.reindex(np.random.permutation(train.index))
train_data = train['Text'].values
train_target = train['Requirement'].values

# Load and randomize test data
test = pd.read_csv(io.StringIO(jobCSVData), header=None, names=['Text','Requirement'])
test = test.reindex(np.random.permutation(test.index))
test_data = test['Text'].values
test_target = test['Requirement'].values

# Get requirements

text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
  ('tfidf', TfidfTransformer(use_idf=True)),
  ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
    alpha=0.001, max_iter=5, random_state=42)),
  ])

text_clf_svm = text_clf_svm.fit(train_data, train_target)
predicted_svm = text_clf_svm.predict(test_data)

i = 0
experience = []
for r in predicted_svm:
    s = test_data[i]
    if r:
        experience.append(s)
    i += 1
    
# Create questions

questions = []
tool = language_check.LanguageTool('en-UK')

for r in experience:
    if len(r.split()) > 2:
        r = r.strip()
        r = r[0].lower() + r[1:]
        q = 'Do you have {0}?'.format(r)
        # Correct any grammatical errors
        matches = tool.check(q)
        nq = language_check.correct(q, matches)
        questions.append(nq)

def app(environ, start_response):
        data = '\n'.join(questions) 
        print(data)
        start_response("200 OK", [
            ("Content-Type", "text/plain"),
            ("Content-Length", str(len(data)))
        ])
        return [bytes(data, 'utf-8')]
