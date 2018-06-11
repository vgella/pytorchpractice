from __future__ import unicode_literals, print_function, division
from io import open
import glob


import unicodedata
import string

all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters)

def uni_to_ascii(s):
   result = ''
   for c in unicodedata.normalize('NFD', s):
      if unicodedata.category(c) != 'Mn' and c in all_letters:
          result += c
   return result
category_lines = {}
all_categories = []

def readLines(filename):
   lines = open(filename, encoding = 'utf-8').read().strip().split('\n')   
   return [uni_to_ascii(line) for line in lines]

for filename in glob.glob('data/names/*.txt'):
   category = filename.split("/")[-1].split(".")[0]
   all_categories += [category]
   category_lines[category] = readLines(filename)
n_categories = len(all_categories)

import torch
def line_to_tensor(line):
   line = line.lower()
   tensor = torch.zeros(len(line), 1, n_letters)
   for i, c in enumerate(line):
      tensor[i][0][all_letters.find(c)] = 1
   return tensor

import random 

def random_example():
   category = random.choice(all_categories)
   name = random.choice(category_lines[category])
   name_tensor = line_to_tensor(name)
   category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
   return category, name, name_tensor, category_tensor
