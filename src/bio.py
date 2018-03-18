from glob import glob
import re
import xmltodict

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


def as_word_list(text):
    l = []
    for word in [word for word in text.lower().split() if word.isalpha() or is_number(word)]:
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



def bioDataGenerator(folder, lang):
    files = glob('{}/*'.format(folder))


    for file in files:
        with open(file) as f:

            text, tags = process_file(f)




if __name__ == '__main__':

    data = bioDataGenerator(lang=English)