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

        self.all_entities = []
        self.acronyms = []
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

    def get_positions(self, tokens, index):
        w = tokens[index][0]
        prev = tokens[index-1][0]
        next = tokens[index+1][0]
        positions = []
        for e in self.all_entities:
            if w in e and prev in e and next in e:
                positions.append(e.index(w))
        return list(set(positions))


    def set_entities(self, entities):
        if entities:
            # dis_list = [l.split() for l in open('../data/disability_tuples.txt', 'r').readlines()]
            #
            # for l in dis_list:
            #     self.all_entities.append([w.lower() for w in l])

            entities = [l.split() for l in entities]

            for l in entities:
                if len(l) == 1 and is_all_caps(l[0]):
                    self.acronyms.append(l[0].lower())
                    # self.all_entities.append([w.lower() for w in l])
                else:
                    self.all_entities.append([w.lower() for w in l])
                    # if len(l) > 1:
                    #     acronym = "".join([w[0] for w in l])
                    #     if acronym not in self.acronyms:
                    #         # self.all_entities.apend(acronym.lower
                    #         # print("Acronym: [{}], Original: [{}]".format(acronym, " ".join(l)))
                    #         self.acronyms.append(acronym.lower())
            # print("Total entities to be written: {}".format(len(self.all_entities)))
            # print(self.all_entities)

            self.all_entities = list(set([tuple(entity) for entity in self.all_entities]))
            self.acronyms = list(set(self.acronyms))

            # for e in self.all_entities:
            #     if len(e) > 1:
            #         acronym = "".join([w[0] for w in e])
            #         if acronym not in self.all_entities:
            #             self.all_entities.append(tuple(acronym))

            with open('../data/entities_{}.txt'.format(self.language), 'w') as f:
                f.write("\n".join([" ".join(line) for line in self.all_entities]))

            with open('../data/acronyms_{}.txt'.format(self.language), 'w') as f:
                f.write("\n".join([" ".join(line) for line in self.all_entities]))
        else:
            with open('../data/entities_{}.txt'.format(self.language), 'r') as f:
                for line in f:
                    self.all_entities.append(line.strip().split())

            with open('../data/acronyms_{}.txt'.format(self.language), 'r') as f:
                for line in f:
                    self.acronyms.append(line.strip())

            self.all_entities = list(set([tuple(entity) for entity in self.all_entities]))
            self.acronyms = list(set(self.acronyms))

        # print("Acronyms: {}".format(self.acronyms))
        # print("Entities: {}".format(self.all_entities))

    def crf_features(self, tokens, index):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous predicted IOB tags
        """

        # init the stemmer
        stemmer = SnowballStemmer(self.language)


        # Pad the sequence with placeholders
        num_of_previous = 3
        num_of_posterior = 2
        tk = []
        for i in range(0, num_of_previous):
            tk.append(('[START{}]'.format(num_of_previous-i), '[START{}]'.format(num_of_previous-i)))

        tk = tk + list(tokens)
        for i in range(1, num_of_posterior+1):
            tk.append(('[END{}]'.format(i), '[END{}]'.format(i)))

        tokens = tk

        index += num_of_previous


        word, pos = tokens[index]


        contains_dash = ('–' in word or '-' in word or '_' in word)
        contains_dot = '.' in word

        prev3_words = tokens[index-3][0] + "_._" + tokens[index-2][0]
        prev3_pos = tokens[index-3][1] + "_._" + tokens[index-2][1]

        prev2_words = tokens[index-2][0] + "_._" + tokens[index-1][0]
        prev2_pos = tokens[index-2][1] + "_._" + tokens[index-1][1]
        # prev2_lemma = stemmer.stem(tokens[index-2][0]) + "_._" + stemmer.stem(tokens[index-1][0])

        prev1_words = tokens[index-1][0] + "_._" + tokens[index][0]
        prev1_pos = tokens[index-1][1] + "_._" + tokens[index][1]
        prev1_lemma = stemmer.stem(tokens[index-1][0]) + "_._" + stemmer.stem(tokens[index][0])

        next1_words = tokens[index][0] + "_._" + tokens[index+1][0]
        next1_pos = tokens[index][1] + "_._" + tokens[index+1][1]

        next2_words = tokens[index+1][0] + "_._" + tokens[index+2][0]
        next2_pos = tokens[index+1][1] + "_._" + tokens[index+2][1]

        allcaps = is_all_caps(word)
        strange_cap = word[0] not in string.ascii_uppercase and word != word.lower()


        # starting_ent = word.lower() in self.starting_entities
        # prev_starting_ent = tokens[index-1][0].lower() in self.starting_entities
        inside_ent = word.lower() in self.all_entities
        is_acronym = word.lower() in self.acronyms
        features = {

            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-caps': allcaps,
            'strange-cap': strange_cap,

            # 'prev3-pos': prev3_pos,
            # 'prev3-word': prev3_words,

            'prev2-pos': prev2_pos,
            'prev2-word': prev2_words,

            'prev1-pos': prev1_pos,
            'prev1-word': prev1_words,
            'prev1-lemma': prev1_lemma,

            'next1-pos': next1_pos,
            'next1-word': next1_words,

            'next2-pos': next2_pos,
            'next2-word': next2_words,



            # 'prev-starting_ent': prev_starting_ent,
            # 'contained_in_ent': inside_ent,  # improves neg but decreases dis
        }
        #
        # # Word features
        # if inside_ent:
        features['inside-entities'] = inside_ent
        positions = self.get_position(word.lower())
        for p in positions:
            features['position-{}'.format(p)] = True
        if is_acronym:
            features['is-acronym'] = is_acronym

        if contains_dash:
            features['contains-dash'] = contains_dash
        if contains_dot:
            features['contains-dot'] = contains_dot


        # # best results were obtained when i added the prev2 attributes with the last results parameters
        #
        for i in range(1, num_of_previous+1):
            word, pos = tokens[index - i]
            lemma = stemmer.stem(word)

            features['prev-{}-word'.format(i)] = word
            features['prev-{}-pos'.format(i)] = pos

            features['prev-{}-lemma'.format(i)] = lemma

        for i in range(1, num_of_posterior + 1):
            word, pos = tokens[index + i]
            inside_ent = word.lower() in self.all_entities

            features['next-{}-word'.format(i)] = word
            features['next-{}-pos'.format(i)] = pos
            features['next-{}-inside-ent'.format(i)] = inside_ent

            # features['next-{}-lemma'.format(num_of_posterior + i)] = lemma # worsens a lot

        #
        #
        #
        #
        # if '(' == tokens[index-1][0]:
        #     features['prev_oparenthesis'] = True
        #
        # # print("Tokens[{}-1] = {}".format(index, tokens[index]))
        # if index+1 < len(tokens) and ')' == tokens[index+1][0]:
        #     features['next_cparenthesis'] = True

        # if is_acronym:
        #     print(features)

        return features

