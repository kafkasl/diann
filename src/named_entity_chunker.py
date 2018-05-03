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
        num_of_previous = 5
        num_of_posterior = 3
        tk = []
        for i in range(0, num_of_previous):
            tk.append(('[START{}]'.format(num_of_previous-i), '[START{}]'.format(num_of_previous-i)))

        tk = tk + list(tokens)
        for i in range(1, num_of_posterior+1):
            tk.append(('[END{}]'.format(i), '[END{}]'.format(i)))

        tokens = tk

        # shift the index with 2, to accommodate the padding
        index += num_of_previous
        # if index == num_of_previous:
        # print("Tokens: {}".format(tokens))
        # print("Index: {} / {}".format(index, len(tokens)))

        word, pos = tokens[index]
        prevword, prevpos = tokens[index - 1]
        prevprevword, prevprevpos = tokens[index - 2]

        contains_dash = ('–' in word or '-' in word or '_' in word)
        contains_dot = '.' in word

        prev2_words = prevprevword + "__" + prevword
        prev2_pos = prevprevpos + "__" + prevpos


        allascii = all([True for c in word if c in string.ascii_lowercase])

        allcaps = is_all_caps(word)
        firstcap = word == word.capitalize()
        capitalized = word[0] in string.ascii_uppercase


        starting_ent = word.lower() in self.starting_word_entities
        inside_ent = word.lower() in self.starting_word_entities + self.inside_word_entities


        # add more or less features to dict
        # example: if the word is inside a seen entity, report the position (or more than one feature if it's present in more than one)
        # this word and the previous/next are inside some entity
        # try to add lemmas in spanish


        features = {

            'word': word,
            'lemma': stemmer.stem(word),
            'pos': pos,
            'all-ascii': allascii,

            'prev2-pos': prev2_pos,
            'prev2-word': prev2_words,
            'starting_ent': starting_ent,
            'contained_in_ent': inside_ent,  # improves neg but decreases dis
        }

        if contains_dash:
            features['contains-dash'] = contains_dash
        if contains_dot:
            features['contains-dot'] = contains_dot

        if allcaps:
            features['all-caps'] = allcaps
        if firstcap:
            features['first-caps'] = firstcap
        if capitalized:
            features['capitalized'] = capitalized



        # best results were obtained when i added the prev2 attributes with the last results parameters

        for i in range(1, num_of_previous+1):
            word, pos = tokens[index - i]
            allcaps = is_all_caps(word)
            firstcap = word == word.capitalize()
            capitalized = word[0] in string.ascii_uppercase
            starting_ent = word.lower() in self.starting_word_entities
            inside_ent = word.lower() in self.starting_word_entities + self.inside_word_entities
            lemma = stemmer.stem(word)

            features['prev-{}-word'.format(num_of_previous-i)] = word
            features['prev-{}-pos'.format(num_of_previous-i)] = pos
            if allcaps:
                features['prev-{}-all-caps'.format(num_of_previous-i)] = allcaps
            if firstcap:
                features['prev-{}-first-cap'.format(num_of_previous-i)] = firstcap
            if capitalized:
                features['prev-{}-capitalized'.format(num_of_previous-i)] = capitalized
            if starting_ent:
                features['prev-{}-starting-ent'.format(num_of_previous-i)] = starting_ent
            # features['prev-{}-inside-ent'.format(num_of_previous-i)] = inside_ent
            features['prev-{}-lemma'.format(num_of_previous-i)] = lemma

        for i in range(1, num_of_posterior + 1):
            word, pos = tokens[index + i]
            allcaps = is_all_caps(word)
            firstcap = prevword == word.capitalize()
            capitalized = word[0] in string.ascii_uppercase
            inside_ent = word.lower() in self.starting_word_entities + self.inside_word_entities
            lemma = stemmer.stem(word)

            features['next-{}-word'.format(num_of_posterior + i)] = word
            features['next-{}-pos'.format(num_of_posterior + i)] = pos
            if allcaps:
                features['next-{}-all-caps'.format(num_of_posterior + i)] = allcaps
            if firstcap:
                features['next-{}-first-cap'.format(num_of_posterior + i)] = firstcap
            if capitalized:
                features['next-{}-capitalized'.format(num_of_posterior + i)] = capitalized
            if inside_ent:
                features['next-{}-inside-ent'.format(num_of_posterior + i)] = inside_ent
            features['next-{}-lemma'.format(num_of_posterior + i)] = lemma

        if contains_dash:
            features['contains-dash'] = contains_dash
        if contains_dot:
            features['contains-dot'] = contains_dot

        if allcaps:
            features['all-caps'] = allcaps
        if firstcap:
            features['first-caps'] = firstcap
        if capitalized:
            features['capitalized'] = capitalized

        if '(' == tokens[index-1][0]:
            features['prev_oparenthesis'] = True

        # print("Tokens[{}-1] = {}".format(index, tokens[index]))
        if index+1 < len(tokens) and ')' == tokens[index+1][0]:
            features['next_cparenthesis'] = True

        return features

