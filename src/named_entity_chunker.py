from collections import Iterable
from nltk.tag import ClassifierBasedTagger, CRFTagger
from nltk.chunk import ChunkParserI
from nltk.stem.snowball import SnowballStemmer

import string
import pickle

def is_number(w):
    try:
        float(w.replace(':', '').replace('’', '').replace(',', '.').replace('–', '').replace('-', '').replace('/', '').replace('.', ''))
    except ValueError:
        return False
    return True

def is_all_caps(w):
    if is_number(w):
        return False
    if '=' in w or '≤' in w or '±' in w:
        return False
    if len(w) == 1:
        return False
    if w != w.upper():
        return False
    return True
    # len(word) > 1 and allcaps and not is_number(word) and word not in ['.', ',', '>', '<', '%', '[', ']', '=', ';',



def iob_features(tokens, index, history):
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




class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents=None, tagger="ClassifierBasedTagger", model=None, model_name="../results/modelCRF_featured", entities=None, language="english", **kwargs):

        self.starting_word_entities = []
        self.inside_word_entities = []
        self.all_entities = []
        self.language = language

        if not model:
            assert isinstance(train_sents, Iterable)

        if tagger == "ClassifierBasedTagger":
            self.feature_detector = iob_features
            self.tagger = ClassifierBasedTagger(
                train=train_sents,
                feature_detector=iob_features,
                **kwargs)

        elif tagger == "CRFTagger":
            self.set_entities(entities)
            if not model:
                # training = []
                # for sentence in train_sents:
                #     s = []
                #     for ((word, pos), tag) in sentence:
                #         s.append((word, tag))
                #     training.append(s)
                # self.tagger = CRFTagger()
                self.tagger = CRFTagger(feature_func=self.crf_features)
                self.tagger.train(train_data=train_sents, model_file="../results/{}".format(model_name))
            else:
                self.tagger = CRFTagger(feature_func=self.crf_features)
                self.tagger.set_model_file(model)
        else:
            raise Exception('Unknown tagger')


    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        # if self.tagger == ClassifierBasedTagger:
            # chunks = [(w, tag) for ((w, pos), tag) in chunks]
            # chunks = [[(w, tag) for ((w, pos), tag) in sentence] for sentence in chunks]
            # iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return chunks
        # return nltk.chunk.conlltags2tree(iob_triplets)

    def get_position(self, w):
        positions = []
        for e in self.all_entities:
            if w in e:
                positions.append(e.index(w))
        return positions

    def set_entities(self, entities):
        if entities:
            dis_list = [l.split() for l in open('../data/disability_tuples.txt', 'r').readlines()]

            self.starting_word_entities = [l[0] for l in dis_list]
            for l in dis_list:
                self.inside_word_entities.extend(l[1:])

            for l in dis_list:
                self.all_entities.append([w.lower() for w in l])

            entities = [l.split() for l in entities]

            self.starting_word_entities.extend([l[0].lower() for l in entities])
            for l in entities:
                self.inside_word_entities.extend([w.lower() for w in l[1:]])
            for l in entities:
                self.all_entities.append([w.lower() for w in l])
            # print("Total entities to be written: {}".format(len(self.all_entities)))
            # print(self.all_entities)

            self.starting_word_entities = list(set(self.starting_word_entities))
            self.inside_word_entities = list(set(self.inside_word_entities))
            self.all_entities = list(set([tuple(entity) for entity in self.all_entities]))

            with open('../data/entities_{}.txt'.format(self.language), 'w') as f:
                f.write("\n".join([" ".join(line) for line in self.all_entities]))
        else:
            with open('../data/entities_{}.txt'.format(self.language), 'r') as f:
                for line in f:
                    self.all_entities.append(line.strip().split())

                self.starting_word_entities.extend([l[0].lower() for l in self.all_entities])
                for l in self.all_entities:
                    self.inside_word_entities.extend([w.lower() for w in l[1:]])

            self.starting_word_entities = list(set(self.starting_word_entities))
            self.inside_word_entities = list(set(self.inside_word_entities))
            self.all_entities = list(set([tuple(entity) for entity in self.all_entities]))





    # def is_followed_by_acronym(self, tokens, distance):
    #     w = tokens[0]

    def crf_features(self, tokens, index):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """

        # init the stemmer
        stemmer = SnowballStemmer(self.language)

        # Pad the sequence with placeholders
        tokens = [('[START3]', '[START3]'), ('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + \
                 [('[END1]', '[END1]'), ('[END2]', '[END2]'), ('[END3]', '[END3]'), ('[END4]', '[END4]'), ('[END5]', '[END5]')]

        # shift the index with 2, to accommodate the padding
        index += 3

        # print("Tokens: {}".format(tokens[index]))

        word, pos = tokens[index]
        prevword, prevpos = tokens[index - 1]
        prevprevword, prevprevpos = tokens[index - 2]
        prevprevprevword, prevprevprevpos = tokens[index - 3]
        nextword, nextpos = tokens[index + 1]
        nextnextword, nextnextpos = tokens[index + 2]
        nextnextnextword, nextnextnextpos = tokens[index + 3]
        contains_dash = ('–' in word or '-' in word or '_' in word)
        contains_dot = '.' in word

        prev2_words = prevprevword + "__" + prevword
        prev2_pos = prevprevpos + "__" + prevpos

        prev_oparenthesis = '(' == prevword
        prev_cparenthesis = ')' == prevword
        next_oparenthesis = '(' == nextword
        next_cparenthesis = ')' == nextword
        allascii = all([True for c in word if c in string.ascii_lowercase])

        allcaps = is_all_caps(word)
        firstcap = word == word.capitalize()
        capitalized = word[0] in string.ascii_uppercase

        prevallcaps = is_all_caps(prevword)
        prevfirstcap = prevword == prevword.capitalize()
        prevcapitalized = prevword[0] in string.ascii_uppercase

        nextallcaps = is_all_caps(nextword)
        nextfirstcap = nextword == nextword.capitalize()
        nextcapitalized = nextword[0] in string.ascii_uppercase

        nnallcaps = nextnextword == nextnextword.capitalize()
        nncapitalized = nextnextword[0] in string.ascii_uppercase

        nnnallcaps = nextnextnextword == nextnextnextword.capitalize()
        nnncapitalized = nextnextnextword[0] in string.ascii_uppercase

        if index + 4 < len(tokens):
            nnnnallcaps = tokens[index+4][0] == tokens[index+4][0].capitalize()
            # print("Token: [{}]".format(tokens[index+4][0]))
            nnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase
        else:
            nnnnallcaps = False
            nnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase


        if len(tokens) > index + 5:
            nnnnnallcaps = tokens[index+5][0] == tokens[index+5][0].capitalize()
            nnnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase
        else:
            nnnnnallcaps = False
            nnnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase


        starting_ent = word.lower() in self.starting_word_entities
        inside_ent = word.lower() in self.inside_word_entities
        in_ent = word.lower() in self.starting_word_entities + self.inside_word_entities

        prev_starting_ent = prevword.lower() in self.starting_word_entities
        prev_ent = prevword.lower() in self.starting_word_entities + self.inside_word_entities
        next_ent = nextword.lower() in self.starting_word_entities + self.inside_word_entities

        contained_in_ent = word.lower() in self.starting_word_entities + self.inside_word_entities

        # contains_y = 'y' in word
        # followed_by_acronym = self.is_followed_by_acronym(tokens[index:])
        # prevprecedent = prevword in self.precedents_list
        #
        # nextprecedent = nextword in precedents_list
        # currentprecedent = word in precedents_list

        # add more or less features to dict
        # example: if the word is inside a seen entity, report the position (or more than one feature if it's present in more than one)
        # this word and the previous/next are inside some entity
        # try to add lemmas in spanish


        # best results were obtained when i added the prev2 attributes with the last results parameters
        features = {
            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': allascii,

            'next-word': nextword,
            'next-lemma': stemmer.stem(nextword),
            'next-pos': nextpos,

            # 'next-next-word': nextnextword,
            # 'next-next-pos': nextnextpos,
            #
            # 'next-next-next-word': nextnextnextword,
            # 'next-next-next-pos': nextnextnextpos,

            'prev-word': prevword,
            'prev-lemma': stemmer.stem(prevword),
            'prev-pos': prevpos,

            # 'prev-prev-word': prevprevword,
            # 'prev-prev-pos': prevprevpos,
            #
            # 'prev-prev-prev-word': prevprevprevword,
            # 'prev-prev-prev-pos': prevprevprevpos,

            'prev2-pos': prev2_pos,
            'prev2-word': prev2_words,

            'starting-eng': starting_ent,
            # 'next-next-capitalized': nncapitalized,
            # 'next-next-next-capitalized': nnncapitalized,
            # 'next-next-next-next-capitalized': nnnncapitalized,
            # 'next-next-next-next-next-capitalized': nnnnncapitalized,
            #
            # 'next-next-all-caps': nnallcaps,
            # 'next-next-next-all-caps': nnnallcaps,
            # 'next-next-next-next-all-caps': nnnnallcaps,
            # 'next-next-next-next-next-all-caps': nnnnnallcaps,
            #
            #
            # 'next-ent': next_ent,  #decreases
            # 'prev-sent': prev_ent, #decreases

            'contained_in_ent': contained_in_ent,


            # 'prev-precedent': prevprecedent,
            # 'next-precedent': nextprecedent,
            # 'current-precedent': currentprecedent

        }
        if contains_dash:
            features['contains-dash'] = contains_dash
        if contains_dot:
            features['contains-dot'] = contains_dot

        if allcaps:
            features['all-caps'] = allcaps
            # features['word-length'] = len(word)
            # print("Word: {}".format(word))
        if firstcap:
            features['first-caps'] = firstcap
        if capitalized:
            features['capitalized'] = capitalized

        # if prevallcaps:
        #     features['prev-all-caps'] = prevallcaps
        # if prevfirstcap:
        #     features['prev-first-caps'] = prevfirstcap
        # if prevcapitalized:
        #     features['prev-capitalized'] = prevcapitalized
        #
        # if nextallcaps:
        #     features['next-all-caps'] = nextallcaps
        # if nextfirstcap:
        #     features['next-first-caps'] = nextfirstcap
        # if nextcapitalized:
        #     features['next-capitalized'] = nextcapitalized
        #

        # if prev_oparenthesis:
        #     features['prev_oparenthesis'] = prev_oparenthesis
        # if prev_cparenthesis:
        #     features['prev_cparenthesis'] = prev_cparenthesis
        # if next_oparenthesis:
        #     features['next_oparenthesis'] = next_oparenthesis
        # if next_cparenthesis:
        #     features['next_cparenthesis'] = next_cparenthesis
        # if starting_ent:
        #     features['starting-ent'] = starting_ent
        # if in_ent:
        #     positions = self.get_position(word)
        #     for i, p in enumerate(positions):
        #         features['inside-ent-{}'.format(i)] = p
        # if prev_starting_ent:
        #     features['prev-starting-sent'] = prev_starting_ent

        return features

