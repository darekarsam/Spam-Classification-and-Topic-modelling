from collections import defaultdict
import operator
import re
from math import log
import random
import os
from os import listdir
from os.path import isfile, join, isdir
from os import walk
import pandas
import filereader as fr

filename = 'distinctive_words.txt'

"""
This function prints the top 10 words associated with each of the 20 topics into the file "distinctive_words.txt".
It is used only in the part2 of this assignment. Input is

all_words - the word count data structure created by train() function
"""
def top10(all_words):
	if os.path.exists(filename):
		os.remove(filename)

	output = open(filename, 'w')
	for topic in all_words:
		output.write('-'*50 + "\n")
		output.write("Top 10 words for " + topic + "\n")
		output.write('-'*50 + "\n")
		top_words = sorted(((k, v) for k,v in all_words[topic].iteritems()), key=lambda tup: tup[1], reverse=True)
		for word in top_words[:10]:
			output.write(str(word[0]) + " - " + str(int(word[1])) + "\n")

	output.close()
	print "Output written to file", filename

"""
This function prints the top 10 words associated with spam and non-spam.
It also prints the last 10 words associated with spam - this is because I found the requirement of part1 a little ambiguous.
The input is

all_words - the word count data structure created by train() function
"""
def spamwords(all_words):
	if 'spam' not in all_words and 'notspam' not in all_words:
		print "This model is not for spam classification"
	else:
		top_words = sorted(((k, v) for k,v in all_words['spam'].iteritems()), key=lambda tup: tup[1], reverse=True)
		print '-'*50
		print "Top 10 words associated with spam"
		print '-'*50
		for word in top_words[:10]:
			print word[0], "-", int(word[1])

		print ""

		print '-'*50
		print "Last 10 words associated with spam"
		print '-'*50
		for word in top_words[-10:]:
			print word[0], "-", int(word[1])

		print ""

		top_words = sorted(((k, v) for k,v in all_words['notspam'].iteritems()), key=lambda tup: tup[1], reverse=True)
		print '-'*50
		print "Top 10 words associated with notspam"
		print '-'*50
		for word in top_words[:10]:
			print word[0], "-", int(word[1])

		print ""

		print '-'*50
		print "Last 10 words associated with notspam"
		print '-'*50
		for word in top_words[-10:]:
			print word[0], "-", int(word[1])

		print ""


def defaultvalue():
	return 0.0000000001

def defaultcollection():
	return defaultdict(defaultvalue)

"""
Function to train the Bayes model. This function is common for the part1 (spam, non-spam) and part2 (topics) case.
For part2 it implements the EM algorithm when the fraction is < 1. The number of iterations for EM are kept to 3 through experimental results for
different values on this dataset. The idea of EM algorithm is pretty simple
E step - assumes the topic assignment for each document is fixed and then calculates the word likelihood ie. P(W|T) over all topics
M step - assumes the word likelihoods are fixed and then calculates the posterior i.e P(T|w1 w2 .. wn) where w1 .. wn belong to a document D.
This is done for all documents D in the train dataset.

The input arguments for it are in the form of 

train_data - this is the data structure returned by the readdata() method in filereader module
fraction - is the fraction input by the user. This is applicable only for part2 and for part1 it takes on its default assignment of 1
part - for which part of the assignment is the function being called viz. "part1" or "part2"
featuretype - can take values of "binary" or "frequency". It mainly controls how the feature vectors are implemented
"""
def train(train_data, part, featuretype="frequency", fraction=1):
	topics = train_data['topics']		
	document_topics = defaultdict(lambda: topics[0])

	model = {}

	iterations = 3 if fraction < 1 else 1
	for i in range(iterations):
		all_words = defaultdict(defaultcollection)

		topic_likelihood = defaultdict(float)		
		topicwise_counts = defaultdict(float)

		# if fraction == 1 no need to run the EM algorithm as we know all the labels
		if fraction < 1:
			print "E step for iteration", i+1 

		
		for doc in train_data['docs']:			
			for word in doc:
				if word != "class_name" and word != "document_name":

					if i == 0:
						# This is the part which does the initial assignment of topics to document based on the value of a randomly generated number 
						# from a uniform probability distribution
						current_topic = random.choice(topics) if random.random() > fraction else doc['class_name']
						document_topics[doc['document_name']] = current_topic
					else:
						current_topic = document_topics[doc['document_name']]

					if featuretype == "frequency":
						topic_likelihood[current_topic] += 1
						all_words[current_topic][word] += 1
						topicwise_counts[current_topic] += 1
					elif featuretype == "binary":
						all_words[current_topic][word] = 1						
		
		if featuretype == "binary":
			for t in topics:		
				topic_likelihood[t] += len(all_words[t])
				topicwise_counts[t] += len(all_words[t])	

		total = sum(topic_likelihood.itervalues(), 0.0)
		topic_likelihood = {k: v / total for k, v in topic_likelihood.iteritems()}

		# if fraction == 1 no need to run the EM algorithm as we know all the labels. Thus the M step is redundant and is skipped
		if fraction < 1:
			correct = 0.0
			total = 0.0	
			
			print "M step for iteration", i+1	
			for doc in train_data['docs']:
				total += 1
				topic_assignment = defaultdict(lambda: 0.0)
				for word in doc:
					if word != "class_name" and word != "document_name":			
						for candidate in topics:												
							topic_assignment[candidate] += log((all_words[candidate][word] / topicwise_counts[candidate]), 2)														

				for topic in topics:
					topic_assignment[topic] += log(topic_likelihood[candidate], 2)

				document_topics[doc['document_name']] = max(topic_assignment.iteritems(), key=operator.itemgetter(1))[0]

				if max(topic_assignment.iteritems(), key=operator.itemgetter(1))[0] == doc['class_name']:
					correct += 1

			print "Accuracy in iteration", i+1, "is", str(int(correct)) + "/" + str(int(total)), str(round(((correct / total) * 100), 2)) + "%"


	top10(all_words) if part == "part2" else spamwords(all_words)

	# final structure contained in the dictionary to be pickled
	model['words'] = all_words	
	model['topic_likelihood'] = topic_likelihood
	model['topicwise_counts'] = topicwise_counts
	model['topics'] = topics
	return model

"""
This function tests the trained Naive Bayes model on the data contained in test_dir.

model - the trained model returned by the train() function
test_dir - directory which contains test documents organized into directories corresponding to actual topic names
"""
def test(model, test_dir, stopwords):	
	topic_likelihood = model['topic_likelihood']
	topics = model['topics']
	all_words = model['words']
	topicwise_counts = model['topicwise_counts']

	print sum(len(v) for k,v in all_words.iteritems()), "words present in the vocabulary"

	confusion_matrix = [[0]*len(topics) for i in range(len(topics))]

	average = 0.0
	average_total = 0.0
	for t in topics:
		correct = 0.0
		total = 0.0		
		for f in listdir(join(os.getcwd(), test_dir, t)):
			total += 1
			topic_assignment = defaultdict(lambda: 0.0)
			for line in open(join(os.getcwd(), test_dir, t, f)):
				for word in line.split():
					word = fr.cleanword(word)

					if word and word not in stopwords:
						for candidate in topics:
							topic_assignment[candidate] += log((all_words[candidate][word] / topicwise_counts[candidate]), 2)

			for topic in topics:
				topic_assignment[topic] += log(topic_likelihood[candidate], 2)
			
			predicted_topic = max(topic_assignment.iteritems(), key=operator.itemgetter(1))[0]
			if predicted_topic == t:
				correct += 1
			confusion_matrix[topics.index(t)][topics.index(predicted_topic)] += 1

		average_total += total
		average += correct
		print "Accuracy for", t, "is", str(int(correct)) + "/" + str(int(total)), str(round(((correct / total) * 100), 2)) + "%"

	print "Average accuracy is", str(round(((average / average_total) * 100), 2)) + "%"

	print "\n", "Here is the confusion matrix"
	matrix = pandas.DataFrame(confusion_matrix, topics, topics)
	print matrix