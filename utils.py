import glob
import os
import random
import unicodedata
from io import open

import torch


class DataPrepare:
    def __init__(self, all_letters) -> None:
        self.all_letters = all_letters
        self.n_letters = len(all_letters)
        self.all_categories = []
        self.category_lines = {}

    def update_attr(self, data_path):
        # "dataset/data/names/*.txt"
        for filename in self.findFiles(data_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn" and c in self.all_letters
        )

    @staticmethod
    def findFiles(path):
        return glob.glob(path)

    def readLines(self, filename):
        lines = open(filename, encoding="utf-8").read().strip().split("\n")
        return [self.unicodeToAscii(line) for line in lines]

    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    def lineToTensor(self, line):
        # why dimension two is 1 ?
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def categoryFromOutput(self, output):
        _, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    @staticmethod
    def randomChoice(line):
        return line[random.randint(0, len(line) - 1)]

    def randomTrainingExample(self):
        # random choose a category
        category = self.randomChoice(self.all_categories)
        # from previous choose category random choose a line from a list
        line = self.randomChoice(self.category_lines[category])
        # get the category tensor
        category_tensor = torch.tensor(
            [self.all_categories.index(category)], dtype=torch.long
        )
        # get the line tensor
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor
