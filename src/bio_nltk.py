from glob import glob
from sklearn.neural_network import MLPClassifier
from nltk.stem.snowball import SnowballStemmer
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from xml.parsers.expat import ExpatError
from collections import defaultdict
from lxml import etree
import numpy as np
from lxml.etree import XMLSyntaxError
from multiprocessing import Pool
from nerc_evaluator import nerc_evaluation
import re
import pickle
import xmltodict
import argparse
import nltk
import json
import string
import sys


def remove_tags(raw):
    """
    Removes the <doc> tags remaining from wikiExtracted data
    :param raw_html: html/text content of a file with many docs
    :return: only text from raw_html
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw)
    return cleantext

def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # init the stemmer
    stemmer = SnowballStemmer('english')

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'),
                                                                                    ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])

    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,

        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,

        'prev-iob': previob,

        'contains-dash': contains_dash,
        'contains-dot': contains_dot,

        'all-caps': allcaps,
        'capitalized': capitalized,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,

        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,

        # word is in a list consult external disabilities
        # create own list from the training
        # be creative
        # try to remove some which may add noise
    }


def is_tag(text, index):
    if text[index] == '<':
        if not (text[index:index+5] in ['<dis>', '<neg>', '<scp>'] or
                    text[index:index+6] in ['</dis>', '</neg>', '</scp>']):
            return False
    if text[index] == '>':
        if not (text[index-5:index+1] in ['<dis>', '<neg>', '<scp>'] or
                    text[index-6:index+1] in ['</dis>', '</neg>', '</scp>']):
            return False

    return True


def remove_unmatched_bracket(text):
    text = list(text)
    i = 0
    while i < len(text):
        if text[i] == '<' or text[i] == '>':
            if not is_tag(text, i):
                del text[i]
        else:
            i += 1

    return "".join(text)


def get_json(raw):
    try:
        node = etree.fromstring('<doc>{}</doc>'.format(raw))
    except XMLSyntaxError as e:
        node = etree.fromstring('<doc>{}</doc>'.format(remove_unmatched_bracket(raw)))

    dictionary = defaultdict(list)

    for e in node.findall('.//dis'):
        dictionary['dis'].extend(nltk.word_tokenize(e.text))

    for e in node.findall('.//neg'):
        dictionary['neg'].extend(nltk.word_tokenize(e.text))

    for e in node.findall('.//scp'):
        aux = []
        for child in e.getchildren():
            aux.extend(nltk.word_tokenize(child.text))
        dictionary['scp'].extend(aux)

    return dictionary


def is_annotated(word, json_tags, parsed_tags=('dis', 'neg')):
    for parsed_tag in parsed_tags:
        try:
            if word in json_tags[parsed_tag]:
                return True, parsed_tag
        except:
            pass
    # try:
    #     if word in json_tags['scp']:
    #         return True
    # except:
    #     pass
    # try:
    #     if word in json_tags['neg']:
    #         return True, 'neg'
    # except:
    #     pass

    return False, ''


def process_sentence(data):

    clean_text = nltk.word_tokenize(remove_tags(data))
    json_tags = get_json(data)

    iob_data = []

    inside = {'dis': False, 'neg': False}
    tagged_text = nltk.pos_tag(clean_text)
    for word, tag in tagged_text:
        annotated, entity = is_annotated(word, json_tags)
        if annotated:
            if inside[entity]:
                iob = 'I-{}'.format(entity)
            else:
                inside = {'dis': False, 'neg': False}
                inside[entity] = True
                iob = 'B-{}'.format(entity)
        else:
            inside = {'dis': False, 'neg': False}
            iob = 'O'
        iob_data.append(((word, tag), iob))

    return iob_data

def bioDataGenerator(folder, lang, debug):
    files = glob('{}/*txt'.format(folder))

    if debug:
        files = files[0:30]

    for file in files:
        print("Processing file: {}".format(file))
        with open(file) as f:
            data = f.read().replace('&', '').replace('\t', '')
            sentences = nltk.sent_tokenize(data)
            total = []
            for s in sentences:
                total.extend(s.split('\n'))
            for sentence in total:
                iob_data = process_sentence(sentence)
                yield iob_data

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, use_neg, **kwargs):
        assert isinstance(train_sents, Iterable)
        training = []

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return iob_triplets
        # return nltk.chunk.conlltags2tree(iob_triplets)

def flatten_to_conll(sentences):
    conll_data = []
    for sentence in sentences:
        if type(sentence) == list:
            for (word, pos), tag in sentence:
                conll_data.append((word, tag))
        else:
            (word, pos), tag = sentence
            conll_data.append((word, tag))
    return conll_data

def write_results_in_conll(words, filename):
    with open(filename, 'w') as f:
        for word, tag in words:
            f.write("{}\t{}\n".format(word, tag))


def process_fold(input):

    fold, training, validation = input

    chunker = NamedEntityChunker(training, use_neg=True)

    validation_results = []
    for sentence in validation:
        result = chunker.parse([(word, pos) for ((word, pos), tag) in sentence])
        for word, pos, tag in result:
            validation_results.append(((word, pos), tag))



    test_data = flatten_to_conll(validation_results)
    gold_data = flatten_to_conll(validation)

    precision, recall = nerc_evaluation(gold_data=gold_data, test_data=test_data)

    write_results_in_conll(flatten_to_conll(validation_results), '../results/validation_results_{}.txt'.format(fold))
    write_results_in_conll(flatten_to_conll(validation), '../results/test_gold_{}.txt'.format(fold))

    return precision, recall


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../data", help="Directory containing input files.")
    parser.add_argument('-l', '--language', default="english", help="Training language")
    parser.add_argument('-d', '--debug', default=False, help="Run in debug mode (just use a single file)")
    parser.add_argument('-f', '--folds', default=10, help="Number of folds for k-fold cross validation")
    parser.add_argument('-t', '--threads', default=4, help="Number of threads to use")

    args = parser.parse_args()

    data_generator = bioDataGenerator(folder=args.input_dir, lang=args.language, debug=args.debug)

    data = list(data_generator)
    chunk_size = len(data)//args.folds
    threads = int(args.threads)

    precisions, recalls = [], []

    inputs = []
    for fold in range(args.folds):
        training = [data[i] for i in
                    [i for i in range(len(data)) if fold * chunk_size > i or i >= (fold + 1) * chunk_size]]
        validation = [data[i] for i in
                      [i for i in range(len(data)) if fold * chunk_size <= i < (fold + 1) * chunk_size]]
        inputs.append((fold, training, validation))


    if threads > 1:
        p = Pool(threads)
        outputs = p.map(process_fold, inputs)
    else:
        outputs = [process_fold(inp) for inp in inputs]

    precisions = [p for p, r in outputs]
    recalls = [r for p, r in outputs]

    print("Total precision: %.2f ± %.2f" % (np.mean(precisions), np.std(precisions)))
    print("Total recall: %.2f ± %.2f" % (np.mean(recalls), np.std(recalls)))




    # score = chunker.evaluate([nltk.chunk.conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
    # print("Chunker.evaluate() score: {}".format(score.accuracy()))
    # sample = "Asthma is not a chronic disease requiring inhaled treatment and in addition " \
    #      "it is a risk factor (RF) of pneumonia."
    # print(chunker.parse(nltk.pos_tag(nltk.word_tokenize(sample))))
