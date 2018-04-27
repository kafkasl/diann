from negex.negex import negTagger, sortRules
from nerc_evaluator import get_entities

import sys
import re


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
            if len(sentence)-1 > i > 0 and w != '.' and w != ',' and sentence[i-1][0] != '(' and \
                            sentence[i - 1][0] != '>' and sentence[i-1][0] != '<' and w != ";" and \
                            w != "%" and w != ':' and w != ')' and sentence[i-1][0] != '?' and w != '?' and \
                            w != ']' and sentence[i-1][0] != '[':
                # print("w: {}, sentence[i-1]: {}".format(w, sentence[i-1]))
                str_sentence += ' '
            if i == 1 and sentence[0][0] == 'Abstract':
                continue
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
        # print("Element: {}".format(elem))

        output.append(tagger.getNegTaggedSentence())

        # if tagger.getNegationFlag() == 'affirmed':
            # print("Sentence: {}".format(sentence))
            # print("Neg tagged: {}".format(tagger.getNegTaggedSentence()))
            # print("Neg flag: {}".format(tagger.getNegationFlag()))

        #     break
    return output

def remove_all_neg_tags(sent):
    while '[PREN]' in sent or '[PSEU]' in sent:
        sent = sent.replace('[PREN]', '')
        sent = sent.replace('[PSEU]', '')
    return sent

def convert_non_negated(sent):
    while '[PHRASE]' in sent:
        sent = sent.replace('[PHRASE]', '<dis>', 1)
        sent = sent.replace('[PHRASE]', '</dis>', 1)
    return sent

def remove_tags_after_neg_phrase(sent):
    s = '[NEGATED]'
    negated_indices = [match.start() for match in re.finditer(re.escape(s), sent)]
    neg_flags_indices = [match.start() for match in re.finditer(re.escape('[PREN]'), sent)]
    neg_flags_indices.extend([match.start() for match in re.finditer(re.escape('[PSEU]'), sent)])

    assert len(neg_flags_indices) % 2 == 0

    s_aux = list(sent)
    remove_idxs = [idx for idx in neg_flags_indices if idx > max(negated_indices)]
    for i in remove_idxs:
        idx = -1
        try:
            idx = sent.find('[PREN]', max(negated_indices))
        except:
            pass
        try:
            idx = sent.find('[PSEU]', max(negated_indices))
        except:
            pass
        if idx == -1:
            break
        else:
            del s_aux[idx:idx+6]
            sent = "".join(s_aux)

    sent = "".join(s_aux)
    return sent

def remove_tags_too_far_from_phrase(sent, distance=3):
    s = '[NEGATED]'
    negated_indices = [match.start() for match in re.finditer(re.escape(s), sent)]
    neg_flags_indices = [match.start() for match in re.finditer(re.escape('[PREN]'), sent)]
    neg_flags_indices.extend([match.start() for match in re.finditer(re.escape('[PSEU]'), sent)])

    # print("sent: {} Indices: {}\nlen: {} ".format(sent, neg_flags_indices, len(neg_flags_indices)))
    if len(neg_flags_indices) % 2 != 0:
        print("EXCEPTION: len {} found".format(len(neg_flags_indices)))

    for i in range(1, len(neg_flags_indices), 2):
        # print("sent: {}\n neg_indices {}\n".format(sent, neg_flags_indices))
        s_aux = list(sent)
        idx0 = neg_flags_indices[i-1]
        idx = neg_flags_indices[i]
        try:
            next_phrase_idx = sent.index(s, idx+6)
        except ValueError:
            return remove_all_neg_tags(sent)
        frag = sent[idx+6:next_phrase_idx]
        words = frag.split()

        neg_sent = "".join(sent[idx0+6:idx])
        # print("neg_sent: {}".format(neg_sent.split()))
        neg_word = neg_sent.split()
        if len(neg_word) > 0:
            neg_word = neg_word[0]
        else:
            neg_word = ""
        # print("Frag: [{}]\nWords: {}\nDistance: {}\nNeg word: {}".format(frag, words, len(words), neg_word))
        if len(words) > distance or (neg_word == 'without' and len(words) > 0):
            del s_aux[idx0:idx0+6]
            del s_aux[idx-6:idx]
            neg_flags_indices = [idx-12 for idx in neg_flags_indices]
        sent = "".join(s_aux)

    return sent


def convert_negated(sent):
    while '[PREN]' in sent:
        # if '[PREN]no signs of[PREN]' in sent:
        #     sent.replace('[PREN]no signs of[PREN]', '<scp><neg>no</neg> signs of ')
        #     sent = sent.replace('[NEGATED]', '<dis>', 1)
        #     sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
        # elif '[PREN]with no signs of[PREN]' in sent:
        #     sent.replace('[PREN]with no signs of[PREN]', '<scp>with <neg>no signs</neg> of ')
        #     sent = sent.replace('[NEGATED]', '<dis>', 1)
        #     sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
        # else:
        sent = sent.replace('[PREN]', '<scp><neg>', 1)
        sent = sent.replace('[PREN]', '</neg>', 1)
        sent = sent.replace('[NEGATED]', '<dis>', 1)
        sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
    while '[PSEU]' in sent:
        if '[PSEU]with' in sent:
                sent = sent.replace('[PSEU]with', '<scp>with <neg>', 1)
                sent = sent.replace('[NEGATED]', '<dis>', 1)
                sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
        elif '[PSEU]presented' in sent:
            sent = sent.replace('[PSEU]presented', '<scp>presented <neg>', 1)
            sent = sent.replace('[NEGATED]', '<dis>', 1)
            sent = sent.replace('[NEGATED]', '</dis></scp>', 1)
        sent = sent.replace('[PSEU]', '</neg>', 1)
    while '[NEGATED]' in sent:
        sent = sent.replace('[NEGATED]', '<dis>', 1)
        sent = sent.replace('[NEGATED]', '</dis>', 1)
    return sent



def convert_into_xml(tagged):
    xml_sent = []
    # print('First sentence: {}'.format(tagged[0]))
    for sent in tagged:
        # Non negated replaced by dis
        sent = convert_non_negated(sent)
        # If there is no negated tag (and non-negated have already been replaced)
        # print("Sentence:\n{}".format(sent))

        if not '[NEGATED]' in sent:
            sent = remove_all_neg_tags(sent)
            # print("Sentence1:\n{}".format(sent))
        # There are negated tags
        else:
            sent = remove_tags_after_neg_phrase(sent)
            # print("Sentence:\n{}".format(sent))
            # print("Sentence1:\n{}".format(sent))
            sent = remove_tags_too_far_from_phrase(sent)
            # print("Sentence2:\n{}".format(sent))
            sent = convert_negated(sent)
            # print("Sentence3:\n{}".format(sent))

        # while '[NEGATED]' in sent or '[PHRASE]' in sent:
        #     
        #     elif '[PHRASE]' in sent:
        #         sent = sent.replace('[PHRASE]', '<dis>', 1)
        #         sent = sent.replace('[PHRASE]', '</dis>', 1)
        # sent = remove_all_neg_tags(sent)
        xml_sent.append(sent)

    return xml_sent

if __name__ == '__main__':

    test = open(sys.argv[1]).readlines()

    data = [l.strip().split('\t') for l in test]
    tagged = find_negated(data=data)
    xml = convert_into_xml(tagged)
    print(xml)
