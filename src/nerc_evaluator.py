import sys



def get_names_entities(lines):
    entities = []
    i = 0
    while i < len(lines):
        word, tag = lines[i]
        entity = []
        if 'B-dis' == tag:
            entity.append(word)
            i += 1
            word, tag = lines[i]
            while 'I-dis' == tag:
                entity.append(word)
                i += 1
                word, tag = lines[i]
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
    gold_entities = get_names_entities(gold_data)
    test_entities = get_names_entities(test_data)
    matching_entities = len([x for x in test_entities if x in gold_entities])

    print("[GOLD] Number of entities: {}".format(len(gold_entities)))
    print("[TEST] Number of entities: {}".format(len(test_entities)))
    print("Test entities matching gold: {}".format(matching_entities))
    print("Accuracy: {}".format(1.0*matching_entities/len(gold_entities)))


# gold = open('test_gold.txt').readlines()
# test = open('test_results.txt').readlines()

if __name__ == '__main__':
    print("Gold reference: {}".format(sys.argv[1]))
    print("Reults: {}".format(sys.argv[2]))

    gold = open(sys.argv[1]).readlines()
    test = open(sys.argv[2]).readlines()

    gold_data = [l.strip().split('\t') for l in gold]
    test_data = [l.strip().split('\t') for l in test]


    gold_entities = get_names_entities(gold_data)
    test_entities = get_names_entities(test_data)
    matching_entities = len([x for x in test_entities if x in gold_entities])

    print("[GOLD] Number of entities: {}".format(len(gold_entities)))
    print("[TEST] Number of entities: {}".format(len(test_entities)))
    print("Test entities matching gold: {}".format(matching_entities))
    print("Accuracy: {}".format(1.0*matching_entities/len(gold_entities)))

#
#
#
# total = 0
# correct = 0
#
# for i in range(0, len(test)):
#     try:
#         wt, tt = test[i].strip().split('\t')
#         wg, tg = gold[i].strip().split('\t')
#         if tg != 'O':
#             total += 1
#             if tt == tg:
#                 correct += 1
#     except ValueError as e:
#         print(e)
#         print("Line {} got: [{}]".format(i, test[i]))
#         print("Line {} got: [{}]".format(i, gold[i]))
#         sys.exit(1)
#
# print("Correct: {}/{}")
# print("Accuracy: {}".format(1.0*correct/total))