import urllib.request
import html2text
import re
import pandas as pd
import numpy as np
import io
import json

import nltk
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import language_check

from flask import Flask
from flask import request
from flask import make_response 

import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger('job-app')
logger.setLevel(logging.INFO)

nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Get job details

train_file_path = 'train-jobs.csv'

def get_job_text(url):
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
        return h.handle(content)

# Format job

def format_sentence(t):
    return "\"{0}\",\"FALSE\"\n".format(t)

def sentence_tokenizer(t):
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
                    result = result + format_sentence(text)
                elif sentence:
                    text = re.sub(r"[\n\r]+", " ", sentence)
                    result = result + format_sentence(text)
    return result

# Create model
def create_model(train_file_path):
    # Load and randomize trainng data
    train = pd.read_csv(train_file_path, header=None, names=['Text','Requirement'])
    train = train.reindex(np.random.permutation(train.index))
    train_data = train['Text'].values
    train_target = train['Requirement'].values

    text_clf_svm = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
      ('tfidf', TfidfTransformer(use_idf=True)),
      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
        alpha=0.001, max_iter=5, random_state=42)),
      ])

    return text_clf_svm.fit(train_data, train_target)
    
# Get requirements

def rephrase(s):
    startPrefix = {"expereince in":"Do you have expereince in",
             "interest in":"Are you interested in",
             "knowledge of":"Do you understand",
             "able to":"Can you",
             "have the":"Do you have the",
             "ability to":"Can you",
             "At least":"Do you have",
                   "must be":"Are you",
                   "should be":"Are you",
                   "should have":"Do you have",
                   "must have":"Do you have"
                  }
    startDefault = "One requierment for the position is "
    
    q = ""
    
    logger.debug(r)
    
    words = r.strip().split()
    if len(words) > 2:
        s = words[0].lower() + ' ' + words[1]
        s1 = startPrefix.get(s, startDefault)
        logger.debug(s)
        logger.debug(s1)
        s2 = r.lower() if s1 == startDefault else r[len(s):]
        logger.debug(s2)
        endPostfix = ". Do you have this" if s1 == startDefault else ""
        q = '{0}{1}{2}?'.format(s1, s2, endPostfix)

    return q

def get_requirements(model,job):
    job_csv_data = sentence_tokenizer(job)

    # Load and randomize test data
    test = pd.read_csv(io.StringIO(job_csv_data), header=None, names=['Text','Requirement'])
    test = test.reindex(np.random.permutation(test.index))
    test_data = test['Text'].values
    test_target = test['Requirement'].values
    predicted_svm = model.predict(test_data)

    # Select requirement sentances
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
            q = rephrase(r) 
            # Correct any grammatical errors
            matches = tool.check(q)
            nq = language_check.correct(q, matches)
            questions.append(nq)

    return questions

@app.route('/', methods=['GET'])
def process():
    data = '{ "Error":"Pass a job url as a query parameter" }'
    url = request.args.get('url')
    resp_code = '400'

    if url and str(url):
        logger.info('URL: {0}',format(url))
        job = get_job_text(url)
        model = create_model(train_file_path)
        questions = get_requirements(model,job)

        data = '\n'.join(json.dumps(questions, indent=4)) 
        resp_code = '200'

    resp = make_response(data,resp_code)
    resp.headers['Content-Type'] = 'application/json' 
    resp.headers['Content-Length'] = str(len(data)) 
    return resp
