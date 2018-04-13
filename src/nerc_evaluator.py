import sys



def get_entities(words):
    entities = []
    i = 0
    while i < len(words):
        word, tag = words[i]
        entity = []
        if 'B-dis' == tag:
            entity.append(word)
            i += 1
            word, tag = words[i]
            while 'I-dis' == tag:
                entity.append(word)
                i += 1
                word, tag = words[i]
            entities.append(" ".join(entity))
        else:
            i += 1

    return entities

def nerc_evaluation(gold_data, test_data):
    """
    Nerc evaluation, format is a list of (word, tag)
    :param gold_data: gold reference
    :param test_data: test reference
    :return: -
    """
    gold_entities = get_entities(gold_data)
    test_entities = get_entities(test_data)
    matching_entities = len([x for x in test_entities if x in gold_entities])

    precision = 1.0*matching_entities/len(test_entities)
    recall = 1.0*matching_entities/len(gold_entities)

    print("[GOLD] Number of entities: {}".format(len(gold_entities)))
    print("[TEST] Number of entities: {}\n".format(len(test_entities)))
    print("Test entities matching gold: {}\n".format(matching_entities))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    return precision, recall


if __name__ == '__main__':
    print("Gold reference: {}".format(sys.argv[1]))
    print("Reults: {}".format(sys.argv[2]))

    gold = open(sys.argv[1]).readlines()
    test = open(sys.argv[2]).readlines()

    gold_data = [l.strip().split('\t') for l in gold]
    test_data = [l.strip().split('\t') for l in test]


    gold_entities = get_entities(gold_data)
    test_entities = get_entities(test_data)
    matching_entities = len([x for x in test_entities if x in gold_entities])

    print("[GOLD] Number of entities: {}".format(len(gold_entities)))
    print("[TEST] Number of entities: {}".format(len(test_entities)))
    print("Test entities matching gold: {}".format(matching_entities))
    print("Accuracy: {}".format(1.0*matching_entities/len(gold_entities)))

