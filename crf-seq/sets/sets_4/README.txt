seq_detect_1.py: 4740 worker spans

seq_detect_2.py: 4740 worker spans + 4740 patterns

seq_detect_3.py: 4740 worker spans + 9480 patterns

seq_detect_4.py: 4740 worker spans + 14192 patterns


do 5 onwards

seq_detect_5.py: 4740*5 worker spans (with duplicates) 

seq_detect_6.py: 4740*5 worker spans (with duplicates) + 4740 patterns

seq_detect_7.py: 4740*5 worker spans (with duplicates) + 9480 patterns

seq_detect_8.py: 4740*5 worker spans (with duplicates) + 14192 patterns


(14192 newly annotated docs: sets of 4740)

old features:
   features = ['bias', 'word.lower=' + word.lower(),'word[-3:]=' + word[-3:],
        'word[-4:]=' + word[-4:],'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(), 'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag, 'isindword=%s' % isindword(word, annotype, indwords_list),
        'word[0:4]=' + word[0:4], 'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)]
    #prev previous word
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][2]

        features.extend(['-1:word.lower=' + word1.lower(), '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list), 'word[0:4]=' + word[0:4],
            'word[-3:]=' + word[-3:]])


new features:

    features = ['bias', 'word.lower=' + word.lower(),'word[-3:]=' + word[-3:],
        'word.islower=%s' % word.islower(), 'word.isupper=%s' % word.isupper(),
        'postag=' + postag, 'isindword=%s' % isindword(word, annotype, indwords_list),
        'word[0:4]=' + word[0:4], 'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)]
    #prev previous word
    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][2]
        features.extend(['-1:word.lower=' + word1.lower(), '-1:postag=' + postag1,
            'isindword=%s' % isindword(word1, annotype, indwords_list),
            'ispattern=%s' % ispattern(word, postag, annotype, pattern_list)])