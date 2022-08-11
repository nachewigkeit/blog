from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
import csv

model = AutoModelForSequenceClassification.from_pretrained('./weight')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')


def tokenize_function(examples):
    return tokenizer(examples["review"], padding='max_length', truncation=True, max_length=274)


dataset = load_dataset("csv", data_files={"train": "dataset/train.csv", "test": "dataset/test.csv"})
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = model.cuda()
correct = []
wrong = []
for i in tokenized_datasets["test"]:
    input_ids = torch.tensor([i['input_ids']]).cuda()
    logits = model(input_ids)['logits'].cpu().detach().numpy()
    predictions = np.argmax(logits, axis=-1)
    if i["label"] == predictions[0]:
        correct.append([i["review"], i["label"]])
    else:
        wrong.append([i["review"], i["label"]])

print("acc: " + str(len(correct) / len(tokenized_datasets["test"])))

with open('correct.csv', 'w', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in correct:
        spamwriter.writerow(row)
with open('wrong.csv', 'w', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in wrong:
        spamwriter.writerow(row)
