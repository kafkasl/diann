from negex.negex import negTagger, sortRules
from nerc_evaluator import get_entities

import sys


def get_entities_per_sentence(words):
    sent_ent = []
    start = 0
    end = 0
    while end < len(words):
        word, tag = words[end]
        if word == '.':
            sentence = words[start:end+1]
            es = get_entities(sentence)
            str_sentence = " ".join([w for w, t in words[start:end+1]])
            sent_ent.append([str_sentence, es])
            start = end
        end += 1

    return sent_ent

def find_negated(data):
    rfile = open(r'./negex/negex_triggers.txt')
    irules = sortRules(rfile.readlines())
    sent_ent = get_entities_per_sentence(data)
    output = []

    for i, elem in enumerate(sent_ent):
        sentence, entities = elem
        # print("Sentence: {}\nEntities: {}\n".format(sentence, entities))
        tagger = negTagger(sentence=sentence, phrases=entities, rules=irules, negP=False)

        # elem.append(tagger.getNegTaggedSentence())
        # elem.append(tagger.getNegationFlag())
        # elem = elem + tagger.getScopes()

        output.append(tagger.getNegTaggedSentence())

        if tagger.getNegationFlag() == 'affirmed':
            # print("Sentence: {}".format(sentence))
            print("Neg tagged: {}".format(tagger.getNegTaggedSentence()))
            # print("Neg flag: {}".format(tagger.getNegationFlag()))

        #     break
    return output


def convert_into_xml(tagged):
    xml_sent = []
    for sent in tagged:
        if '[NEGATED]' in sent:
            sent = sent.replace('[PREN]', '<scp><neg>', 1)
            sent = sent.replace('[PREN]', '</neg>', 1)
            sent = sent.replace('[NEGATED]', '<dis>', 1)
            sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
            xml_sent.append(sent)
        if '[PHRASE]' in sent:
            sent = sent.replace('[PHRASE]', '<dis>', 1)
            sent = sent.replace('[PHRASE]', '</dis>', 1)
            xml_sent.append(sent)


    print(xml_sent)

if __name__ == '__main__':

    test = open(sys.argv[1]).readlines()

    data = [l.strip().split('\t') for l in test]
    tagged = find_negated(data=data)
    convert_into_xml(tagged)
