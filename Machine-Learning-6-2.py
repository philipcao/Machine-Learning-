# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:02:56 2019

@author: Yuan
"""

import re
import numpy as np

def process_text(text):
    dot_word = r'。|，|\.|\?|,|!|，|\(|\)|\(|\)| '
    return re.sub(dot_word, '', text)

def text2vec(text):
    cleaned_text = process_text(text)
    text_vec = cleaned_text.split()
    vocab_list = list(set(text_vec))
    numer_vec = [0.]*len(vocab_list)
    for word in text_vec:
        numer_vec[vocab_list.index(word)] += 1
    return numer_vec / np.sum(numer_vec)

text = 'Without a doubt, a loving and friendly puppy or dog can put an instant smile on your face! When you adopt a dog from Atlanta Humane Society, you gain a wonderful canine companion. But most of all, when you adopt a rescue dog, you have the ability to bond with one of Atlanta’s forgotten and neglected animals.'

text2vec(text)

