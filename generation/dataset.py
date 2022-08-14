import csv

data = [["text"]]
num = 0
with open("dataset/raw.txt", encoding='utf-8') as file:
    for row in file:
        row = row.strip()
        if "更新时间" not in row and len(row) > 0:
            data.append([row])
            num += len(row)

print(num)

with open('dataset/train.csv', 'w', newline='', encoding="utf-8") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for row in data:
        spamwriter.writerow(row)
