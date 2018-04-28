from glob import glob
from glob import glob
from collections import defaultdict
from lxml import etree
from lxml.etree import XMLSyntaxError
from multiprocessing import Pool
from nerc_evaluator import *
from named_entity_chunker import NamedEntityChunker
from neg_finder import find_negated, convert_into_xml

import numpy as np

import re
import argparse
import nltk
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

def bioDataGenerator(files, lang):

    for file in files:
        # print("Processing file: {}".format(file))
        with open(file) as f:
            data = f.read().replace('&', '').replace('\t', '')
            total = [s.strip() for s in data.split('\n')]
            for i, s in enumerate(total):
                if len(s) == 0 or total[i][-1] != '.':
                    total[i] += '.'
            if total[-1] == '.':
                del total[-1]
            # sentences = nltk.sent_tokenize(data)
            # total = []
            # for s in sentences:
            #     processed = s.split('\n')
            #     if len(processed) > 1:
            #         for i in range(0, (len(processed))):
            #             if processed[i][-1] != '.':
            #                 processed[i] += '.'
            #     total.extend(processed)
            for sentence in total:
                iob_data = process_sentence(sentence)
                yield iob_data


def flatten_to_conll(sentences, contains_pos=False):
    conll_data = []
    # print("Setnences: {}".format(sentences))
    for sentence in sentences:
        s_aux = []
        if type(sentence) == list:
            for word, tag in sentence:
                if contains_pos:
                    word, _ = word
                s_aux.append((word, tag))
        else:
            word, tag = sentence
            if contains_pos: word, _ = word
            s_aux.append((word, tag))
        conll_data.append(s_aux)
    return conll_data


# def write_results_in_conll(words, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     with open(folder+filename, 'w') as f:
#         for word, tag in words:
#             f.write("{}\t{}\n".format(word, tag))


def predict(chunker, validation):

    validation_results = []

    for sentence in validation:
        s_aux = []
        result = chunker.parse([(word, pos) for ((word, pos), tag) in sentence])
        for (word, pos), tag in result:
            s_aux.append((word, tag))
        validation_results.append(s_aux)

    return validation_results


def process_fold(input):

    fold, training_files, gold_files, provided, tagger, model_name, language = input

    if args.debug:
        training_files, gold_files = training_files[0:2], gold_files[0:1]
        print("Input files length:\nTraining: {}\nGold: {}\n".format(len(training_files), len(gold_files)))
        print("Input files:\nTraining: {}\nGold: {}\n".format(training_files, gold_files))

    training_data = list(bioDataGenerator(files=training_files, lang=args.language))
    # print("training data: {}".format(training_data))
    tr = []
    for sentence in training_data:
        for ((word, pos), tag) in sentence:
            tr.append((word, tag))

    entities = get_entities(tr)
    # print("Entities: {}".format(entities))
    if not provided:
        chunker = NamedEntityChunker(train_sents=training_data, tagger=tagger,
                                     model_name=model_name, entities=entities, language=language)
    else:
        chunker = NamedEntityChunker(tagger=tagger, model=model_name, entities=entities, language=language)

    if gold_files:
        test(tagger=tagger, gold_files=gold_files, chunker=chunker, language=language)
    else:
        test(tagger=tagger, gold_files=training_files, chunker=chunker, language=language)


def test(tagger, gold_files, chunker, language):
    predictions = {}
    for file in gold_files:
        print("Predicting file: {}".format(file))
        validation_data = list(bioDataGenerator(files=[file], lang=language))
        prediction = predict(chunker=chunker, validation=validation_data)
        predictions[file] = prediction

    # gold = list(bioDataGenerator(files=gold_files, lang=args.language))
    # system = predict(tagger, chunker=chunker, validation=gold)
    #
    # system_data = flatten_to_conll(system)
    # gold_data = flatten_to_conll(gold, contains_pos=True)
    #
    # precision, recall = nerc_evaluation(gold_data=gold_data, test_data=system_data)
    #
    # system_conll = flatten_to_conll(system)
    # gold_conll = flatten_to_conll(gold)
    #
    # write_results_in_conll(system_conll, '../results/system/conll/', 'test_{}.txt'.format(fold))
    # write_results_in_conll(gold_conll, '../results/gold/conll/', 'test_{}.txt'.format(fold))

    for file, prediction in predictions.items():
        pred_conll = flatten_to_conll(prediction)
        tagged = find_negated(data=pred_conll, language=language)
        xml = convert_into_xml(tagged)
        filename = '../results/system/{}/{}/{}'.format(tagger, language, file.split("/")[-1])
        with open(filename, 'w') as f:
            f.write("\n".join(xml))
            print("Written results in {}".format(filename))


            # return precision, recall



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../data", help="Directory containing input files.")
    parser.add_argument('-l', '--language', default="english", help="Training language")
    parser.add_argument('-d', '--debug', default=False, help="Run in debug mode (just use a single file)", type=bool)
    parser.add_argument('-f', '--folds', default=10, help="Number of folds for k-fold cross validation", type=int)
    parser.add_argument('-n', '--number_of_threads', default=4, help="Number of threads to use", type=int)
    parser.add_argument('-m', '--model_name', default=None, help="Trained model files for CRF")
    parser.add_argument('-p', '--provided_model', action='store_true')
    parser.add_argument('-t', '--tagger', default="CRFTagger", help="Tagger to use:  CRFTagger or ClassifierBasedTagger")
    parser.add_argument('--testing', action='store_true', default=False)

    args = parser.parse_args()

    files = glob('{}/*txt'.format(args.input_dir))
    chunk_size = len(files)//args.folds
    threads = int(args.number_of_threads)
    model_name = args.model_name
    provided_model = args.provided_model


    if not args.testing:
        inputs = []
        if args.folds > 1:
            for fold in range(args.folds):
                model = "{}_{}".format(model_name, fold)
                training = [files[i] for i in
                            [i for i in range(len(files)) if fold * chunk_size > i or i >= (fold + 1) * chunk_size]]
                validation = [files[i] for i in
                              [i for i in range(len(files)) if fold * chunk_size <= i < (fold + 1) * chunk_size]]
                inputs.append((fold, training, validation, provided_model, args.tagger, model, args.language))
                print("Training size: {}\nValidation size: {}".format(len(training), len(validation)))
        else:
            model = "{}_final".format(model_name)
            print("Training final model: {} with {} elements".format(model, len(files)))
            inputs = [(0, files, [], provided_model, args.tagger, model, args.language)]
        # print("Inputs: {}".format(inputs))

        if threads > 1:
            p = Pool(threads)
            p.map(process_fold, inputs)
        else:
            for inp in inputs:
                process_fold(inp)

    else:
        chunker = NamedEntityChunker(tagger=args.tagger, model=model_name, entities=None, language=args.language)
        test(tagger=args.tagger, gold_files=files, chunker=chunker, language=args.language)


    # precisions = [p for p, r in outputs]
    # recalls = [r for p, r in outputs]
    #
    # print("Total precision: %.2f ± %.2f" % (np.mean(precisions), np.std(precisions)))
    # print("Total recall: %.2f ± %.2f" % (np.mean(recalls), np.std(recalls)))




    # score = chunker.evaluate([nltk.chunk.conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
    # print("Chunker.evaluate() score: {}".format(score.accuracy()))
    # sample = "Asthma is not a chronic disease requiring inhaled treatment and in addition " \
    #      "it is a risk factor (RF) of pneumonia."
    # print(chunker.parse(nltk.pos_tag(nltk.word_tokenize(sample))))
