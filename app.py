import urllib.request
import html2text
import re
import pandas as pd
import numpy as np
import io

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
    grammar = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    sentance = s.split()
    tagged_s = pos_tag(sentance)

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tagged_s)
    term_noun = ''

    for n in result.leaves():
        if n[1] == 'NN' or n[1] == 'NNP':
            term_noun = n[0]

    # Create map

    prefix = { 'experience in' : 'Do you have experience in',
        'must have' : 'Do you have',
        'ability to' : 'Can you',
        'able to' : 'Can you',
        'proficient in' : 'Are you proficient in'
    }

    w = s.lower().split()
    p = '{0} {1}'.format(w[0],w[1])
    p2 = prefix[p] 

    ns = '{0} {1}'.format(p2, s[len(p) + 1:])
    i = ns.find(term_noun) + len(term_noun)
    q = '{0}?'.format(ns[:i])

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
    data = 'Error: Pass a job url as a query parameter\n'
    url = request.args.get('url')
    resp_code = '400'

    print('URL:',url)

    if str(url):
        job = get_job_text(url)
        model = create_model(train_file_path)
        questions = get_requirements(model,job)

        data = '\n'.join(questions) 
        resp_code = '200'

    resp = make_response(data,resp_code)
    resp.headers['Content-Type'] = 'text/plain' 
    resp.headers['Content-Length'] = str(len(data)) 
    return resp
