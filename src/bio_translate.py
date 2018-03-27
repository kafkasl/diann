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

import re
import pickle
import xmltodict
import argparse
import nltk
import json
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


def is_annotated(word, json_tags):
    try:
        if word in json_tags['dis']:
            return True, 'dis'
    except:
        pass
    # try:
    #     if word in json_tags['scp']:
    #         return True
    # except:
    #     pass
    try:
        if word in json_tags['neg']:
            return True, 'neg'
    except:
        pass

    return False, ''


def process_sentence(data):

    clean_text = nltk.word_tokenize(remove_tags(data))
    json_tags = get_json(data)

    iob_data = []

    inside = False
    tagged_text = nltk.pos_tag(clean_text)
    for word, tag in tagged_text:
        annotated, tag = is_annotated(word, json_tags)
        if annotated:
            if inside:
                iob = 'I-{}'.format(tag)
            else:
                inside = True
                iob = 'B-{}'.format(tag)
        else:
            inside = False
            iob = 'O'
        iob_data.append(((word, tag), iob))

    return iob_data

def bioDataGenerator(folder, lang, debug):
    files = glob('{}/*txt'.format(folder))

    if debug:
        files = [files[0]]

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
                for line in iob_data:
                    yield line


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../data", help="Directory containing input files.")
    parser.add_argument('-l', '--language', default="english", help="Training language")
    parser.add_argument('-d', '--debug', default=False, help="Run in debug mode (just use a single file)")

    args = parser.parse_args()

    data_generator = bioDataGenerator(folder=args.input_dir, lang=args.language, debug=args.debug)

    data = list(data_generator)
    print("Number of words: {}".format(len(data)))

    with open('../data/English_Training/english_iob.txt', 'w') as f:
        for elem in data:
            f.write("{}\t{}\n".format(elem[0][0], elem[1]))
