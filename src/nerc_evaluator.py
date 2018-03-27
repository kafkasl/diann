
gold = open('test_gold.txt').readlines()
test = open('test_results.txt').readlines()

for i in range(0, len(test)):
    wt, tt = test[i].split('\t')
    wg, tg = gold[i].split('\t')
    if tg != 'O':
        total += 1
        if tt == tg:
            correct += 1
