# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:45:03 2019

@author: Yuan
"""

from os import listdir
from os.path import isfile, join
import jieba
import codecs
from langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pickle
import random

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

# Helper Function

def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff
	
# Get List of Files

dataBaseDirPos = "./data/ChnSentiCorp_htl_ba_6000/pos/"
dataBaseDirNeg = "./data/ChnSentiCorp_htl_ba_6000/neg/"
positiveFiles = [dataBaseDirPos + f for f in listdir(dataBaseDirPos) if isfile(join(dataBaseDirPos, f))]
negativeFiles = [dataBaseDirNeg + f for f in listdir(dataBaseDirNeg) if isfile(join(dataBaseDirNeg, f))]


# Show length of Sample
print(len(positiveFiles))
print(len(negativeFiles))

# Look at what's in files
filename = positiveFiles[0]
with codecs.open(filename, "r") as doc_file:
    text=doc_file.read()
    print(text)
	
# Removing stop words
filename = positiveFiles[110]
with codecs.open(filename, "r") as doc_file:
    text=doc_file.read()
    text = text.replace("\n", "")
    text = text.replace("\r", "")
print("==Orginal==:\n\r{}".format(text))
    
stopwords = [ line.rstrip() for line in codecs.open('./data/chinese_stop_words.txt',"r", encoding="utf-8") ]
seg_list = jieba.cut(text, cut_all=False)
final =[]
seg_list = list(seg_list)
for seg in seg_list:
    if seg not in stopwords:
        final.append(seg)
print("==Tokenized==\tToken count:{}\n\r{}".format(len(seg_list)," ".join(seg_list)))
print("==Stop Words Removed==\tToken count:{}\n\r{}".format(len(final)," ".join(final)))

# Prepare documents
documents = []
for filename in positiveFiles:
    text = ""
    with codecs.open(filename, "rb") as doc_file:
        for line in doc_file:
            try:
                line = line.decode("GB2312")
            except:
                continue
            text+=Converter('zh-hans').convert(line)# Convert from traditional to simplified Chinese

            text = text.replace("\n", "")
            text = text.replace("\r", "")
    documents.append((text, "pos"))

for filename in negativeFiles:
    text = ""
    with codecs.open(filename, "rb") as doc_file:
        for line in doc_file:
            try:
                line = line.decode("GB2312")
            except:
                continue
            text+=Converter('zh-hans').convert(line)# Convert from traditional to simplified Chinese

            text = text.replace("\n", "")
            text = text.replace("\r", "")
    documents.append((text, "neg"))

# Save and load files
# Uncomment those two lines to save/load the documents for later use since the step above takes a while
# __pickleStuff("./data/chinese_sentiment_corpus.p", documents)
# documents = __loadStuff("./data/chinese_sentiment_corpus.p")
print(len(documents))
print(documents[4000])

# shaffle the data
random.shuffle(documents)

# Prepare the input and output for the models
# Tokenize only
totalX = []
totalY = [str(doc[1]) for doc in documents]
for doc in documents:
    seg_list = jieba.cut(doc[0], cut_all=False)
    seg_list = list(seg_list)
    totalX.append(seg_list)

#Switch to below code to experiment with removing stop words
# Tokenize and remove stop words
# totalX = []
# totalY = [str(doc[1]) for doc in documents]
# stopwords = [ line.rstrip() for line in codecs.open('./data/chinese_stop_words.txt',"r", encoding="utf-8") ]
# for doc in documents:
#     seg_list = jieba.cut(doc[0], cut_all=False)
#     seg_list = list(seg_list)
#     Uncomment below code to experiment with removing stop words
#     final =[]
#     for seg in seg_list:
#         if seg not in stopwords:
#             final.append(seg)
#     totalX.append(final)

#Visualize 
import numpy as np
import scipy.stats as stats
import pylab as pl
h = sorted([len(sentence) for sentence in totalX])
maxLength = h[int(len(h) * 0.60)]
print("Max length is: ",h[len(h)-1])
print("60% cover length up to: ",maxLength)
h = h[:5000]
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-o')
pl.hist(h,normed=True)      #use this to draw histogram of your data
pl.show() 

# Words to number token
totalX = [" ".join(wordslist) for wordslist in totalX]  # Keras Tokenizer expect the words tokens to be seperated by space 
input_tokenizer = Tokenizer(30000) # Initial vocab size
input_tokenizer.fit_on_texts(totalX)
vocab_size = len(input_tokenizer.word_index) + 1
print("input vocab_size:",vocab_size)
totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))
__pickleStuff("./data/input_tokenizer_chinese.p", input_tokenizer)

# Output array
target_tokenizer = Tokenizer(3)
target_tokenizer.fit_on_texts(totalY)
print("output vocab_size:",len(target_tokenizer.word_index) + 1)
totalY = np.array(target_tokenizer.texts_to_sequences(totalY)) -1
totalY = totalY.reshape(totalY.shape[0])

# turn output
totalY = to_categorical(totalY, num_classes=2)
totalY[40:50]
output_dimen = totalY.shape[1] # which is 2

# save models
target_reverse_word_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
sentiment_tag = [target_reverse_word_index[1],target_reverse_word_index[2]] 
metaData = {"maxLength":maxLength,"vocab_size":vocab_size,"output_dimen":output_dimen,"sentiment_tag":sentiment_tag}
__pickleStuff("./data/meta_sentiment_chinese.p", metaData)

# build model and train
embedding_dim = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim,input_length = maxLength))
# Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
# All the intermediate outputs are collected and then passed on to the second GRU layer.
model.add(GRU(256, dropout=0.9, return_sequences=True))
# Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
model.add(GRU(256, dropout=0.9))
# The output is then sent to a fully connected layer that would give us our final output_dim classes
model.add(Dense(output_dimen, activation='softmax'))
# We use the adam optimizer instead of standard SGD since it converges much faster
tbCallBack = TensorBoard(log_dir='./Graph/sentiment_chinese', histogram_freq=0,
                            write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(totalX, totalY, validation_split=0.1, batch_size=32, epochs=20, verbose=1, callbacks=[tbCallBack])
model.save('./data/sentiment_chinese_model.HDF5')

print("Saved model!")

# prediction
model = None
sentiment_tag = None
maxLength = None
def loadModel():
    global model, sentiment_tag, maxLength
    metaData = __loadStuff("./data/meta_sentiment_chinese.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    sentiment_tag = metaData.get("sentiment_tag")
    embedding_dim = 256
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(GRU(256, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        model.add(Dense(output_dimen, activation='softmax'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('./data/sentiment_chinese_model.HDF5')
        model.summary()
    print("Model weights loaded!")

	
# Functions to convert sentence to model input, and predict result
def findFeatures(text):
    text=Converter('zh-hans').convert(text)
    text = text.replace("\n", "")
    text = text.replace("\r", "") 
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = list(seg_list)
    text = " ".join(seg_list)
    textArray = [text]
    input_tokenizer_load = __loadStuff("./data/input_tokenizer_chinese.p")
    textArray = np.array(pad_sequences(input_tokenizer_load.texts_to_sequences(textArray), maxlen=maxLength))
    return textArray
def predictResult(text):
    if model is None:
        print("Please run \"loadModel\" first.")
        return None
    features = findFeatures(text)
    predicted = model.predict(features)[0] # we have only one sentence to predict, so take index 0
    predicted = np.array(predicted)
    probab = predicted.max()
    predition = sentiment_tag[predicted.argmax()]
    return predition, probab
	
# Calling the load model function
loadModel()
