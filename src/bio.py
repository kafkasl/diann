from glob import glob
from sklearn.neural_network import MLPClassifier

import numpy as np
import re
import xmltodict
import argparse
import nltk

scp_open = '<scp>'
scp_close = '</scp>'
neg_open = '<neg>'
neg_close = '</neg>'
dis_open = '<dis>'
dis_close = '</dis>'

xml_tags = [scp_open, scp_close, neg_open, neg_close, dis_open, dis_close]

def is_number(s):
    """
    Checks if the parameter s is a number
    :param s: anything
    :return: true if s is a number, false otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def remove_tags(raw):
    """
    Removes the <doc> tags remaining from wikiExtracted data
    :param raw_html: html/text content of a file with many docs
    :return: only text from raw_html
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw)
    return cleantext


def as_word_list(text, only_words=False, to_lower=True):
    l = []
    text = text.split()
    if to_lower:
        text = [word.lower() for word in text]
    if only_words:
        text = [word for word in text if word.isalpha() or is_number(word)]
    for word in text:
        l.append(word)
    return l


def get_json(raw):
    dictionary = xmltodict.parse('<doc>{}</doc>'.format(raw))['doc']
    for k in dictionary.keys():
        if type(dictionary[k]) != list:
            dictionary[k] = as_word_list(dictionary[k])
        else:
            clean = []
            for lst in dictionary[k]:
                clean.extend(as_word_list(lst))
            dictionary[k] = clean

    return dictionary

def process_file(f):


    data = f.read()
    raw_text = data.replace('\r', ' ').replace('\n', ' ')
    clean_text = as_word_list(remove_tags(raw_text))
    json_tags = get_json(raw_text)

    bio_data = []


    inside = False
    for word in clean_text:
        if word in json_tags['dis']:
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


def process_nltk_file(f):


    data = f.read()
    raw_text = data.replace('\r', ' ').replace('\n', ' ')
    clean_text = nltk.word_tokenize(remove_tags(raw_text))
    json_tags = get_json(raw_text)

    bio_data = []


    inside = False
    for word in clean_text:
        if word in json_tags['dis']:
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

    bio_data = []
    for file in files:
        print("Processing file: {}".format(file))
        with open(file) as f:

            bio_data.append(process_file(f))

    X = [v for k, v in bio_data]
    y = [k for k, v in bio_data]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    clf.fit(X, y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../data", help="Directory containing input files.")
    parser.add_argument('-l', '--language', default="english", help="Training language")
    parser.add_argument('-d', '--debug', default=False, help="Run in debug mode (just use a single file)")

    args = parser.parse_args()

    data = bioDataGenerator(folder=args.input_dir, lang=args.language, debug=args.debug)