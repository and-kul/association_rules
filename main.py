import pandas as pd
import numpy as np
import scipy
import json


import Orange
from orangecontrib.associate.fpgrowth import frequent_itemsets, association_rules, rules_stats

from_id_to_name = dict()

with open("data/features.txt", "r") as file:
    for line in file:
        words = line.split()
        from_id_to_name[int(words[0])] = " ".join(words[1:])


files = [
    "data/atlanta.txt",
    "data/boston.txt",
    "data/chicago.txt",
    "data/los_angeles.txt",
    "data/new_orleans.txt",
    "data/new_york.txt",
    "data/san_francisco.txt",
    "data/washington_dc.txt",
]

transactions = []

for filepath in files:
    with open(filepath, "r") as file:
        for line in file:
            columns = line.split("\t")
            features = [int(x) for x in columns[2].split(" ")]
            transactions.append(features)

transactions_count = len(transactions)
print(transactions_count)


itemsets_dict = dict(frequent_itemsets(transactions, min_support=100))

rules = association_rules(itemsets_dict, min_confidence=0.75)

stats = list(rules_stats(rules, itemsets_dict, transactions_count))

stats.sort(key=lambda t: t[6], reverse=True)

print(len(stats))



with open("results.txt", "w") as results_file:
    for t in stats:
        left = [from_id_to_name[id] for id in t[0]]
        right = [from_id_to_name[id] for id in t[1]]
        support = t[2]
        confidence = t[3]
        lift = t[6]

        results_file.write("{0} -> {1} (support = {2}, confidence = {3:.2f}, lift = {4:.2f})\n".format(left, right, support,
                                                                                              confidence, lift))



