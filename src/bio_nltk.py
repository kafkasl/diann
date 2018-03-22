from glob import glob
from sklearn.neural_network import MLPClassifier
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.stem.snowball import SnowballStemmer
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI


import numpy as np

import re
import pickle
import xmltodict
import argparse
import nltk
import string
import os



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


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):  # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]



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
    }


def get_json(raw):
    dictionary = xmltodict.parse('<doc>{}</doc>'.format(raw))['doc']
    if type(dictionary) == dict:
        for k in dictionary.keys():
            if type(dictionary[k]) != list:
                dictionary[k] = nltk.word_tokenize(dictionary[k])
            else:
                clean = []
                for lst in dictionary[k]:
                    clean.extend(nltk.word_tokenize(lst))
                dictionary[k] = clean
    else:
        dictionary = {}
    return dictionary


def is_annotated(word, json_tags):
    annotated = False
    try:
        if word in json_tags['dis']:
            annotated = True
    except:
        pass
    try:
        if word in json_tags['scp']:
            annotated = True
    except:
        pass
    try:
        if word in json_tags['neg']:
            annotated = True
    except:
        pass

    return annotated

def process_sentence(data):

    #data = data.replace('\r', ' ').replace('\n', ' ')
    clean_text = nltk.word_tokenize(remove_tags(data))
    json_tags = get_json(data)

    bio_data = []

    inside = False
    for word in clean_text:
        if is_annotated(word, json_tags):
        # if word in json_tags['dis'] or word in json_tags['scp'] or word in json_tags['neg']:
            if inside:
                tag = 'I'
            else:
                inside = True
                tag = 'B'
        else:
            inside = False
            tag = 'O'
        bio_data.append((word, tag))

    return bio_data

def bioDataGenerator(folder, lang, debug):
    files = glob('{}/*txt'.format(folder))

    # print("Files found {}".format(files))
    if debug:
        files = [files[0]]

    for file in files:
        print("Processing file: {}".format(file))
        with open(file) as f:
            for sentence in nltk.sent_tokenize(f.read()):
                bio_data = process_sentence(sentence)
                yield bio_data

            # X = [v for k, v in bio_data]
            # y = [k for k, v in bio_data]
            # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
            # clf.fit(X, y)


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)

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

        return iob_triplets

        # Transform the list of triplets to nltk.Tree format
        # return conlltags2tree(iob_triplets)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../data", help="Directory containing input files.")
    parser.add_argument('-l', '--language', default="english", help="Training language")
    parser.add_argument('-d', '--debug', default=False, help="Run in debug mode (just use a single file)")

    args = parser.parse_args()

    data_generator = bioDataGenerator(folder=args.input_dir, lang=args.language, debug=args.debug)
    corpus_root = '../data/gmb-1.0.0'


    # try:
    #     while True:
    #         print(next(data_generator))
    #         print('------------')
    # except StopIteration as e:
    #     print("Exception, list should be empty: {}".format(e))
    #

    print("List is empty")
    # reader = read_gmb(corpus_root)
    data = list(data_generator)
    training_samples = data[:int(len(data) * 0.9)]
    test_samples = data[int(len(data) * 0.9):]

    print("#training samples = %s" % len(training_samples))  # training samples = 55809
    print("#test samples = %s" % len(test_samples))  # test samples = 6201

    chunker = NamedEntityChunker(training_samples[:2000])

    sample = "Asthma is a chronic disease requiring inhaled treatment and in addition " \
             "it is a risk factor (RF) of pneumonia."
    print(chunker.parse(nltk.pos_tag(nltk.word_tokenize(sample))))

    # score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
    # print(score.accuracy())
