from collections import Iterable
from nltk.tag import ClassifierBasedTagger, CRFTagger
from nltk.chunk import ChunkParserI
from nltk.stem.snowball import SnowballStemmer

import string
import pickle

dis_list = [l.split() for l in open('../data/disability_tuples.txt', 'r').readlines()]

starters = [l[0] for l in dis_list]
insiders = []
for l in dis_list:
    insiders.extend(l[1:])

precedents_list = [l.strip() for l in open('../data/precedents.txt', 'r').readlines()]



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
    def __init__(self, train_sents=None, tagger="ClassifierBasedTagger", model=None, model_name="../results/modelCRF_featured", entities=None, **kwargs):

        self.starting_word_entities = []
        self.inside_word_entities = []
        self.all_entities = []

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

    def set_entities(self, entities):
        entities = [l.split() for l in entities]
        if entities:
            self.starting_word_entities = [l[0].lower() for l in entities]
            self.inside_word_entities = []
            for l in entities:
                self.inside_word_entities.extend([w.lower() for w in l[1:]])
            self.all_entities = []
            for l in entities:
                self.all_entities.extend([w.lower() for w in l])

            # print("Entities: {}".format(self.all_entities))



    # def is_followed_by_acronym(self, tokens, distance):
    #     w = tokens[0]

    def crf_features(self, tokens, index):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """

        # init the stemmer
        stemmer = SnowballStemmer('english')

        # Pad the sequence with placeholders
        tokens = [('[START3]', '[START3]'), ('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) +\
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
        contains_dash = '-' in word
        contains_dot = '.' in word

        prev2_words = prevprevword + "__" + prevword
        prev2_pos = prevprevpos + "__" + prevpos

        # prev_oparenthesis = '(' == prevword
        # prev_cparenthesis = ')' == prevword
        # next_oparenthesis = '(' == nextword
        # next_cparenthesis = ')' == nextword
        allascii = all([True for c in word if c in string.ascii_lowercase])

        allcaps = word == word.upper()
        firstcap = word == word.capitalize()
        capitalized = word[0] in string.ascii_uppercase

        prevallcaps = prevword == prevword.upper()
        prevfirstcap = prevword == prevword.capitalize()
        prevcapitalized = prevword[0] in string.ascii_uppercase

        nextallcaps = nextword == nextword.upper()
        nextfirstcap = nextword == nextword.capitalize()
        nextcapitalized = nextword[0] in string.ascii_uppercase

        # nnallcaps = nextnextword == nextnextword.capitalize()
        # nncapitalized = nextnextword[0] in string.ascii_uppercase
        #
        # nnnallcaps = nextnextnextword == nextnextnextword.capitalize()
        # nnncapitalized = nextnextnextword[0] in string.ascii_uppercase
        #
        # if index + 4 < len(tokens):
        #     nnnnallcaps = tokens[index+4][0] == tokens[index+4][0].capitalize()
        #     # print("Token: [{}]".format(tokens[index+4][0]))
        #     nnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase
        # else:
        #     nnnnallcaps = False
        #     nnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase
        #
        #
        # if len(tokens) > index + 5:
        #     nnnnnallcaps = tokens[index+5][0] == tokens[index+5][0].capitalize()
        #     nnnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase
        # else:
        #     nnnnnallcaps = False
        #     nnnnncapitalized = tokens[index+4][0][0] in string.ascii_uppercase

        starting_dis = word.lower() in starters
        inside_dis = word.lower() in insiders

        starting_ent = word.lower() in self.starting_word_entities
        inside_ent = word.lower() in self.inside_word_entities

        prev_starting_ent = prevword.lower() in self.starting_word_entities
        prev_ent = prevword.lower() in self.all_entities
        next_ent = nextword.lower() in self.all_entities

        # contained_in_ent = word in self.all_entities

        # contains_y = 'y' in word
        # followed_by_acronym = self.is_followed_by_acronym(tokens[index:])
        # prevprecedent = prevword in precedents_list

        # nextprecedent = nextword in precedents_list
        # currentprecedent = word in precedents_list

        # add more or lesss features to dict
        # example: if the word is inside a seen entity, report the position (or more than one feature if it's present in more than one)
        # this word and the previous/next are inside some entity
        # try to add lemmas in spanish

        return {
            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': allascii,

            'next-word': nextword,
            'next-lemma': stemmer.stem(nextword),
            'next-pos': nextpos,

            'next-next-word': nextnextword,
            'next-next-pos': nextnextpos,

            'next-next-next-word': nextnextnextword,
            'next-next-next-pos': nextnextnextpos,

            'prev-word': prevword,
            'prev-lemma': stemmer.stem(prevword),
            'prev-pos': prevpos,

            'prev-prev-word': prevprevword,
            'prev-prev-pos': prevprevpos,

            'prev-prev-prev-word': prevprevprevword,
            'prev-prev-prev-pos': prevprevprevpos,

            'prev2-pos': prev2_pos,
            'prev2-word': prev2_words,

            'contains-dash': contains_dash,
            'contains-dot': contains_dot,

            'all-caps': allcaps,
            'first-caps': firstcap,
            'capitalized': capitalized,

            'prev-all-caps': prevallcaps,
            'prev-first-caps': prevfirstcap,
            'prev-capitalized': prevcapitalized,

            'next-all-caps': nextallcaps,
            'next-first-caps': nextfirstcap,
            'next-capitalized': nextcapitalized,
            # 'next-next-capitalized': nncapitalized,
            # 'next-next-next-capitalized': nnncapitalized,
            # 'next-next-next-next-capitalized': nnnncapitalized,
            # 'next-next-next-next-next-capitalized': nnnnncapitalized,
            #
            # 'next-next-all-caps': nnallcaps,
            # 'next-next-next-all-caps': nnnallcaps,
            # 'next-next-next-next-all-caps': nnnnallcaps,
            # 'next-next-next-next-next-all-caps': nnnnnallcaps,

            'starting_dis': starting_dis,
            'inside_dis': inside_dis,

            'starting_ent': starting_ent,
            'inside_ent': inside_ent,  # improves neg but decreases dis
            'prev-starting-sent': prev_starting_ent,

            # 'next-ent': next_ent,  #decreases
            # 'prev-sent': prev_ent, #decreases

            # 'contained_in_ent': contained_in_ent,
            # 'prev_oparenthesis': prev_oparenthesis,
            # 'prev_cparenthesis ': prev_cparenthesis,
            # 'next_oparenthesis': next_oparenthesis,
            # 'next_cparenthesis': next_cparenthesis,
            #
            # 'prev-precedent': prevprecedent,
            # 'next-precedent': nextprecedent,
            # 'current-precedent': currentprecedent,



            # word is in a list consult external disabilities
            # create own list from the training
            # be creative
            # try to remove some which may add noise
        }

