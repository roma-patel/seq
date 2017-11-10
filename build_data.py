from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import json, re

path = '/nlp/data/romap/set/'
path = '/Users/romapatel/Desktop/set/'

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """

    # get config and processing of words
    # loads PubMeda articles
    config = Config(load=False)
    print 'Config'
    processing_word = get_processing_word(lowercase=True)
    print 'Processing_word'

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    print 'Loaded dev, test, train'

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    print 'Loading vocab_words'
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)


if __name__ == "__main__":
    main()

    #loading data
    '''train, test, dev = [], [], []
    f = open(path + 'data/docids/gold.txt', 'r')
    for line in f: test.append(line.strip())

    f = open(path + 'data/docids/dev.txt', 'r')
    for line in f: dev.append(line.strip())
    
    f = open(path + 'data/docids/train.txt', 'r')
    for line in f: train.append(line.strip())
    
    f = open(path + 'data/annotations/HMMCrowd/training_all.json', 'r')
    for line in f:
        anno_dict = json.loads(line)


    #test
    f = open('/Users/romapatel/Desktop/lstm-crf/data/test.txt', 'w+')
    f_2 = open('/Users/romapatel/Desktop/lstm-crf/data/test_docids.txt', 'w+')
    for i in range(len(test)):
        docid = test[i]
        print docid
        f_2.write(docid + '\n')
        f.write('-DOCSTART- -X- O O\n\n')
        sentences = anno_dict[docid]
        for sentence in sentences:
            for word_tuple in sentence:
                if re.search('\n', word_tuple[0]) is not None: continue
                tag = 'N'
                if word_tuple[5] == 1: tag = 'O'
                if word_tuple[4] == 1: tag = 'I'
                if word_tuple[3] == 1: tag = 'P'
                f.write(word_tuple[0].encode('utf-8') + ' ' + word_tuple[2].encode('utf-8') + ' ' + tag + '\n')
            f.write('\n')

    #train
    f = open('/Users/romapatel/Desktop/lstm-crf/data/dev.txt', 'w+')
    f_2 = open('/Users/romapatel/Desktop/lstm-crf/data/dev_docids.txt', 'w+')
    for i in range(len(dev)):
        docid = dev[i]
        print docid
        f_2.write(docid + '\n')
        f.write('-DOCSTART- -X- O O\n\n')

        sentences = anno_dict[docid]
        for sentence in sentences:
            for word_tuple in sentence:
                if re.search('\n', word_tuple[0]) is not None: continue
                tag = 'N'
                if word_tuple[5] == 1: tag = 'O'
                if word_tuple[4] == 1: tag = 'I'
                if word_tuple[3] == 1: tag = 'P'
                f.write(word_tuple[0].encode('utf-8') + ' ' + word_tuple[2].encode('utf-8') + ' ' + tag + '\n')
            f.write('\n')
    
    #train
    f = open('/Users/romapatel/Desktop/lstm-crf/data/train.txt', 'w+')
    f_2 = open('/Users/romapatel/Desktop/lstm-crf/data/train_docids.txt', 'w+')

    for i in range(len(train)):
        docid = train[i]
        print docid
        if docid not in anno_dict.keys(): continue
        f_2.write(docid + '\n')
        f.write('-DOCSTART- -X- O O\n\n')

        sentences = anno_dict[docid]
        for sentence in sentences:
            for word_tuple in sentence:
                if re.search('\n', word_tuple[0]) is not None: continue
                tag = 'N'
                if word_tuple[5] == 1: tag = 'O'
                if word_tuple[4] == 1: tag = 'I'
                if word_tuple[3] == 1: tag = 'P'
                f.write(word_tuple[0].encode('utf-8') + ' ' + word_tuple[2].encode('utf-8') + ' ' + tag + '\n')
            f.write('\n')'''
