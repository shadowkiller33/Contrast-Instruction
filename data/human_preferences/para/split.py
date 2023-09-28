#split a json file into three parts: test and train and dev
import json
import random
import os
import argparse

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

random.shuffle(data)
train = data[:int(len(data)*0.7)]
dev = data[int(len(data)*0.7):int(len(data)*0.8)]
test = data[int(len(data)*0.8):]

print(len(train))
print(len(dev))
print(len(test))

with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=4)
with open('dev.json', 'w', encoding='utf-8') as f:
    json.dump(dev, f, ensure_ascii=False, indent=4)
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)