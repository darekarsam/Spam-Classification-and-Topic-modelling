import sys
import os
from os import listdir
from os.path import isfile, join, isdir
from os import walk
from collections import defaultdict
import operator
import re
from math import log
import random

"""
Uses regular expressions to search for a proper match within a noisy token. The match regex is straightforward, 
but is powerful when removing garbage characters present in a token. It discards a token completely if a search for this regex fails.
"""
def cleanword(word):
	word = word.strip().lower()
	
	m = re.search(r"\w+.*\w+", word)
	val = m.group(0) if m else None

	return val

"""
Function to read in the stopwords from the stopwords.txt file.
It returns a dictionary which is used when reading in the data and testing on the documents.
"""
def stopwords():
	stopwords = {}
	for w in open('stopwords.txt'):
		w = w.strip()
		if len(w) > 0:
			stopwords[w] = 1
	return stopwords

"""
Used to tokenize the data and eliminate the stopwords when building the data structure essential for naive bayes classifier.
It only stores the counts, they are later converted into probabilities. The inputs are

directory - the input directory holding the data. The documents are expected to be organized in folders whose names correspond to labels for all documents within them.
stopwords - the dictionary of words to ignore when building the model's data structure.
"""
def readdata(directory, stopwords):
	stopwords_eliminated = 0
	
	print "Started reading the documents"
	subdirectories = [f for f in listdir(join(os.getcwd(), directory)) if isdir(join(os.getcwd(), directory, f))]
	data = {}
	data['docs'] = []
	data['topics'] = subdirectories
	for topic in subdirectories:
		for f in listdir(join(os.getcwd(), directory, topic)):
			wordvector = defaultdict(float)
			for line in open(join(os.getcwd(), directory, topic, f)):
				for word in line.split():
					word = cleanword(word)

					if word:
						if word not in stopwords:
							wordvector[word] += 1
						else:
							stopwords_eliminated +=1

			wordvector['class_name'] = topic
			wordvector['document_name'] = join(topic, f)
			data['docs'].append(wordvector)			

	print "Finished reading the documents"
	print "Stop words eliminated were", stopwords_eliminated
	return data
