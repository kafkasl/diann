from negex.negex import negTagger, sortRules
from nerc_evaluator import get_entities

import sys


# def get_entities_per_sentence(words):
#     sent_ent = []
#     start = 0
#     end = 0
#     while end < len(words):
#         word, tag = words[end]
#         if word == '.':
#             sentence = words[start:end+1]
#             es = list(set(get_entities(sentence)))
#             str_sentence = " ".join([w for w, t in words[start:end+1]])
#             sent_ent.append([str_sentence, es])
#             start = end + 1
#         end += 1
#
#     return sent_ent

def get_entities_per_sentence(sentences):
    sent_ent = []

    for sentence in sentences:
        es = list(set(get_entities(sentence)))
        str_sentence = ""
        for i, (w, _) in enumerate(sentence):
            if w != '.' and len(sentence)-1 > i > 0:
                str_sentence += ' '
            str_sentence += w

        sent_ent.append([str_sentence, es])

    return sent_ent

def find_negated(data):
    # rfile = open(r'../src/negex/negex_triggers.txt')
    rfile = open(r'../src/negex/custom_negex_triggers.txt')
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

        # if tagger.getNegationFlag() == 'affirmed':
            # print("Sentence: {}".format(sentence))
            # print("Neg tagged: {}".format(tagger.getNegTaggedSentence()))
            # print("Neg flag: {}".format(tagger.getNegationFlag()))

        #     break
    return output

def remove_all_neg_tags(sent):
    sent = sent.replace('[PREN]', '')
    sent = sent.replace('[PSEU', '')
    return sent

def convert_into_xml(tagged):
    xml_sent = []
    # print('First sentence: {}'.format(tagged[0]))
    for sent in tagged:
        while '[NEGATED]' in sent or '[PHRASE]' in sent:
            if '[NEGATED]' in sent:
                if '[PREN]':
                    sent = sent.replace('[PREN]', '<scp><neg>', 1)
                    sent = sent.replace('[PREN]', '</neg>', 1)
                elif '[PSEU]' in sent:
                    if '[PSEU]with' in sent:
                            sent = sent.replace('[PSEU]with', '<scp>with <neg>', 1)
                    elif '[PSEU]presented' in sent:
                        sent = sent.replace('[PSEU]presented', '<scp>presented <neg>', 1)
                    sent = sent.replace('[PSEU]', '</neg>', 1)
                sent = sent.replace('[NEGATED]', '<dis>', 1)
                sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
            elif '[PHRASE]' in sent:
                sent = sent.replace('[PHRASE]', '<dis>', 1)
                sent = sent.replace('[PHRASE]', '</dis>', 1)
            else:
                sent = remove_all_neg_tags(sent)
        xml_sent.append(sent)

    return xml_sent

if __name__ == '__main__':

    test = open(sys.argv[1]).readlines()

    data = [l.strip().split('\t') for l in test]
    tagged = find_negated(data=data)
    xml = convert_into_xml(tagged)
    print(xml)
