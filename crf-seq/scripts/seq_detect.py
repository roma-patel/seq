import pycrfsuite
import sklearn
from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import re
import json

path = '/nlp/data/romap/crf/'
annotypes = ['Participants', 'Intervention', 'Outcome']
annotype = annotypes[0]

def run():
    train_sents, test_sents = get_train_test_sets()
    indwords_list = get_ind_words()
    X_train = [sent_features(sent, indwords_list) for sent in train_sents]
    y_train = [sent_labels(sent) for sent in train_sents]

    X_test = [sent_features(sent, indwords_list) for sent in test_sents]
    y_test = [sent_labels(sent) for sent in test_sents]
    trainer = pycrfsuite.Trainer(verbose=False)
    
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({'c1': 1.0,'c2': 1e-3, 'max_iterations': 50, 'feature.possible_transitions': True})
    trainer.train('PICO.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open('PICO.crfsuite')

    f = open(path + annotype + '-test.txt', 'w+')
    for i in range(0, 9):
        f.write('New docid:\n')
        print_results(test_sents[i], tagger, indwords_list, f)

def get_ind_words():
    fin_list = []
    for annotype in annotypes:
        list = []
        filename = annotype.lower() + '_words.txt'
        f = open('/Users/romapatel/Desktop/' + filename, 'r')
        for line in f:
            word = line[:-1]
            if word not in list:
                list.append(word)
        fin_list.append(list)
    indwords = [fin_list[0], fin_list[1], fin_list[2]]
    return indwords
    
def isindword(word, annotype, indwords_list):
    if annotype == annotypes[0]:
        list = indwords_list[0]
    elif annotype == annotypes[1]:
        list = indwords_list[1]
    else:
        list = indwords_list[2]

    '''for item in list:
        if item in word.lower(): return True
    return False'''
    if word.lower() in list or word.lower()[:-1] in list or word.lower()[-3:] in list: return True
    else: return False

def word_features(sent, i, indwords_list):
    word = sent[i][0]
    postag = sent[i][2]
    features = ['bias', 'word.lower=' + word.lower(),'word[-3:]=' + word[-3:],
        'word[-4:]=' + word[-4:],'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(), 'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag, 'isindword=%s' % isindword(word, annotype, indwords_list),
                'word[0:4]=' + word[0:4]]
    #prev previous word
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][2]

        features.extend(['-1:word.lower=' + word1.lower(), '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
                         'word[-3:]=' + word[-3:]])
    #previous word
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][2]

        features.extend(['-1:word.lower=' + word1.lower(), '-1:word.istitle=%s' % word1.istitle(),

            '-1:word.isupper=%s' % word1.isupper(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
                         'word[-3:]=' + word[-3:]])
    else:
        features.append('BOS')
    #next to next word
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][2]

        features.extend(['+1:word.lower=' + word1.lower(), '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(), '+1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
                         'word[-3:]=' + word[-3:]])
    #next word
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][2]

        features.extend(['+1:word.lower=' + word1.lower(), '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(), '+1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
                         'word[-3:]=' + word[-3:]])
    else:
        features.append('EOS')
    return features

def sent_features(sent, indwords_list):
    return [word_features(sent, i, indwords_list) for i in range(len(sent))]

def sent_labels(sent):
    return [str(p_label) for token, ner, postag, p_label, i_label, o_label in sent]

def sent_tokens(sent):
    return [token for token, ner, postag, p_label, i_label, o_label in sent]

def print_results(example_sent, tagger, indwords_list, f):
    pred, correct = tagger.tag(sent_features(example_sent, indwords_list)), sent_labels(example_sent)
    spans, span, outside = [], [], True
    for i in range(len(pred)):
        if pred[i] == '0' and outside is True: continue
        elif pred[i] == '0' and outside is False:
            span.append(i+1)
            spans.append(span)
            span, outside = [], True
        elif pred[i] == '1' and outside is False: continue
        elif pred[i] == '1' and outside is True:
            outside = False
            span.append(i)
            
    f.write('\n\nPredicted: ' + str(spans) + '\n')
    for span in spans:
        s = ' '
        for i in range(span[0], span[1]):
            s += example_sent[i][0] + ' '
        f.write(s + '\n')

    spans, span, outside = [], [], True
    for i in range(len(correct)):
        if correct[i] == '0' and outside is True: continue
        elif correct[i] == '0' and outside is False:
            span.append(i+1)
            spans.append(span)
            span, outside = [], True
        elif correct[i] == '1' and outside is False: continue
        elif correct[i] == '1' and outside is True:
            outside = False
            span.append(i)
    f.write('\n\nCorrect: ' + str(spans) + '\n')
    for span in spans:
        s = ' '
        for i in range(span[0], span[1]):
            s += example_sent[i][0] + ' '
        f.wriite(s + '\n')

def evaluate():
    labels = list(crf.classes_)
    labels.remove('0')
    print labels
    
    
#dict[docid]['gt']['Participants'] = [[word,ner,pos,p_label,i_label,o_label], .. ,[]]
def get_training_data():
    #f = open('/Users/romapatel/Desktop/difficulty_crf.json', 'r')
    f = open(path + 'difficulty_crf_mv.json', 'r')

    for line in f:
        dict = json.loads(line)
    return dict

def get_train_test_sets():
    #test_docids = '23549581 6882809 12780755 18824288 21304253 20576092 3475309 21418794 17089419 22511135'
    test_docids = '12452403 24726342 24215858 23688137 16316486 25675062 9070546 10078673 10078672 16767966'
    test_docids = test_docids.split(' ')
    doc_dict = get_training_data()
    test_sents, train_sents = [], []
    count = 0
    for docid in doc_dict:
        #sents = doc_dict[docid]['gt']['Participants']
        sents = doc_dict[docid]['mv']['Participants']
        if len(sents) == 0: continue
        count += 1
        if count >= 300: break
        '''if count <= 131:
            train_sents.append(sents)
        if count > 131:
            test_sents.append(sents)'''
        if docid not in test_docids:
            train_sents.append(sents)
        else:
            test_sents.append(sents)
    return train_sents, test_sents
    
if __name__ == '__main__':
    run()
    #evaluate()

