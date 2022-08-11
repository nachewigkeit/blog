import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

raw = []
stat = {'num': [0, 0], 'length': [[], []]}
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
with open("dataset/ChnSentiCorp_htl_all.csv", newline='', encoding="utf-8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if len(row[-1]) >= 1:
            raw.append(row)
            tokens = tokenizer(row[-1])['input_ids']
            if row[0] == "0":
                stat['num'][0] += 1
                stat['length'][0].append(len(tokens))
            elif row[0] == "1":
                stat['num'][1] += 1
                stat['length'][1].append(len(tokens))
raw = raw[1:]
print("num", stat['num'][0], ',', stat['num'][1])

plt.hist(stat['length'][0] + stat['length'][1], bins=100)
plt.xlabel("Number of Tokens")
plt.show()
len0 = np.array(stat['length'][0])
len1 = np.array(stat['length'][1])
print("mean:", np.mean(len0), ',', np.mean(len1))
print("std:", np.std(len0), ',', np.std(len1))

print("long:", np.sum(len0 > 274) + np.sum(len1 > 274))
'''
random.shuffle(raw)
rate = 0.8
length = int(len(raw) * rate)
train = [['label', 'review']]
test = [['label', 'review']]
train += raw[:length]
test += raw[length:]

with open('dataset/train.csv', 'w', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in train:
        spamwriter.writerow(row)

with open('dataset/test.csv', 'w', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in test:
        spamwriter.writerow(row)
'''
