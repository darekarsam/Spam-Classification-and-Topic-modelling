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
import pickle

# This list contains attributes which have already been used in the decision tree and cannot be used again for splitting on.
cantuselist = []

# This is the data structure for the decision tree nodes
# data contains the word or will contain boolean True/False for leaf nodes
# True indicates a spam Document and False indicates a non-spam document
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None # Word or True/False
        self.isLeaf = False

# This method is used to print the decision tree given root node.
def print_tree(rootnode):
  thislevel = [rootnode]
  while thislevel:
    nextlevel = list()
    for n in thislevel:
      print n.data,
      if n.left: nextlevel.append(n.left)
      if n.right: nextlevel.append(n.right)
    print
    thislevel = nextlevel

# This method is used to print the decision tree given root node.
def print_tree2(rootnode):
    if rootnode is None:
        return
    print "Current Node is : " + str(rootnode.data)
    try:
        print " Left of " + str(rootnode.data) + " is " + str(rootnode.left.data)
    except AttributeError:
        print " Left of " + str(rootnode.data) + " is "
    try:
        print " Right of " + str(rootnode.data) + " is " + str(rootnode.right.data)
    except AttributeError:
        print " Right of " + str(rootnode.data) + " is "
    print_tree2(rootnode.left)
    print_tree2(rootnode.right)

# This method is used to print the top 4 levels of decision tree given root node.
def print_tree3(rootnode):
    print "\nThe top 4 levels of Decision Tree are as follows:"
    # Level 1
    if rootnode is not None and rootnode.data is not None:
        print "\t\t\t\t" + str(rootnode.data) + "\n"
    # Level 2
    if rootnode is not None and rootnode.left is not None and rootnode.left.data is not None:
        print "\t\t\t" + str(rootnode.left.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.data is not None:
        print "\t\t\t\t" + str(rootnode.right.data) + "\n"
    # Level 3
    if rootnode is not None and rootnode.left is not None and rootnode.left.left is not None and rootnode.left.left.data is not None:
        print "\t" + str(rootnode.left.left.data),
    if rootnode is not None and rootnode.left is not None and rootnode.left.right is not None and rootnode.left.right.data is not None:
        print "\t\t\t\t" + str(rootnode.left.right.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.left is not None and rootnode.right.left.data is not None:
        print "\t\t\t\t\t" + str(rootnode.right.left.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.right is not None and rootnode.right.right.data is not None:
        print "\t\t\t" + str(rootnode.right.right.data) + "\n"
    # Level 4
    if rootnode is not None and rootnode.left is not None and rootnode.left.left is not None and rootnode.left.left.left is not None and rootnode.left.left.left.data is not None:
        print "" + str(rootnode.left.left.left.data),
    if rootnode is not None and rootnode.left is not None and rootnode.left.left is not None and rootnode.left.left.right is not None and rootnode.left.left.right.data is not None:
        print "\t\t" + str(rootnode.left.left.right.data),
    if rootnode is not None and rootnode.left is not None and rootnode.left.right is not None and rootnode.left.right.left is not None and rootnode.left.right.left.data is not None:
        print "\t\t" + str(rootnode.left.right.left.data),
    if rootnode is not None and rootnode.left is not None and rootnode.left.right is not None and rootnode.left.right.right is not None and rootnode.left.right.right.data is not None:
        print "\t\t" + str(rootnode.left.right.right.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.left is not None and rootnode.right.left.left is not None and rootnode.right.left.left.data is not None:
        print "\t" + str(rootnode.right.left.left.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.left is not None and  rootnode.right.left.right is not None and rootnode.right.left.right.data is not None:
        print "\t\t" + str(rootnode.right.left.right.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.right is not None and rootnode.right.right.left is not None and rootnode.right.right.left.data is not None:
        print "\t\t\t\t\t\t\t" + str(rootnode.right.right.left.data),
    if rootnode is not None and rootnode.right is not None and rootnode.right.right is not None and rootnode.right.right.right is not None and rootnode.right.right.right.data is not None:
        print "\t\t\t\t" + str(rootnode.right.right.right.data) + "\n"

def mylog(x,base):
    if x == 0:
    	return 0
    else:
    	return log(x,base)

def mydiv(num,den):
    if num == 0:
        return 0
    else:
        return (1.0*num)/(1.0*den)

# This method checks if word is present in doc
def isPresent(doc,word):
    return True if doc.get(word)>0 else False

# This method reads the documents and stores them in a dictionary
def readData(directory):
    words = {}
    wordcount = {}
    subdirectories = [f for f in listdir(join(os.getcwd(), directory)) if isdir(join(os.getcwd(), directory, f))]
    data = {}
    for topic in subdirectories:
        data[topic] = {}
        wordcount[topic] = {}
        for f in listdir(join(os.getcwd(), directory, topic)):
            wordvector = defaultdict(float)
            for line in open(join(os.getcwd(), directory, topic, f)):
                for word in line.split():
                    word = word.strip().lower()
                    if wordvector[word] == 0.0:
                        if word not in wordcount[topic]:
                            wordcount[topic][word] = 1
                        else:
                            wordcount[topic][word] += 1
                    wordvector[word] += 1
                    words[word] = 1
            data[topic][f] = wordvector
    return (data,words,wordcount)

# This method is recursively called to split the dataset and get the word with minimum avg disorder (entropy)
def generateDecisionTree(trained_data,words,wordcount):
    if len(trained_data['notspam'])==0:
        node = Tree()
        node.data = True
        node.isLeaf = True
        return node
    if len(trained_data['spam'])==0:
        node = Tree()
        node.data = False
        node.isLeaf = True
        return node
    minpred = 9999
    minpredword = ''
    for word in words:
        havingword = wordcount['notspam'].get(word,0) + wordcount['spam'].get(word,0)
        total = len(trained_data['spam']) + len(trained_data['notspam'])
        nothavingword = total - havingword
        spamhavingword = wordcount['spam'].get(word,0)
        spamnothavingword = len(trained_data['spam']) - spamhavingword
        notspamhavingword  = wordcount['notspam'].get(word,0)
        notspamnothavingword = len(trained_data['notspam']) - notspamhavingword
        avgdisorder = \
        (mydiv(havingword,total))* \
        (-(mydiv(spamhavingword,havingword))*mylog(mydiv(spamhavingword,havingword),2) \
        - (mydiv(notspamhavingword,havingword))*mylog(mydiv(notspamhavingword,havingword),2)) \
        + \
        (mydiv(nothavingword,total))* \
        (- (mydiv(spamnothavingword,nothavingword))*mylog(mydiv(spamnothavingword,nothavingword),2) \
         - (mydiv(notspamnothavingword,nothavingword))*mylog(mydiv(notspamnothavingword,nothavingword),2))
        # print "Total Number of Documents: " + str(total)
        # print "Number of Documents Containing Word: " + word + " " +str(havingword)
        # print "Out of those spam are: " + str(spamhavingword)
        # print "Out of those not spam are: " + str(notspamhavingword)
        # print "\nNumber of Documents Not Containing Word: " + word + " " +str(nothavingword)
        # print "Out of those spam are: " + str(spamnothavingword)
        # print "Out of those not spam are: " + str(notspamnothavingword)
        # print "Average Disorder is: " + str(avgdisorder)
        if  minpred > avgdisorder:
            minpred = avgdisorder
            minpredword = word

    # New Split Up Dataset and Counts
    new_trained_data_without_word = {}
    new_trained_data_without_word['spam'] = {}
    new_trained_data_without_word['notspam'] = {}

    new_trained_data_with_word = {}
    new_trained_data_with_word['spam'] = {}
    new_trained_data_with_word['notspam'] = {}

    wordcount_with = {}
    wordcount_with['spam'] = {}
    wordcount_with['notspam'] = {}
    wordcount_without = {}
    wordcount_without['spam'] = {}
    wordcount_without['notspam'] = {}

    wordlist_with = {}
    wordlist_without = {}

    for doc in trained_data['spam']:
        if trained_data['spam'][doc].get(minpredword)<1:
            new_trained_data_without_word['spam'][doc] = trained_data['spam'][doc]
            for word in trained_data['spam'][doc]:
                wordlist_without[word]=1
                if word not in wordcount_without['spam']:
                    wordcount_without['spam'][word] = 1
                else:
                    wordcount_without['spam'][word] += 1
        else:
            new_trained_data_with_word['spam'][doc] = trained_data['spam'][doc]
            for word in trained_data['spam'][doc]:
                wordlist_with[word]=1
                if word not in wordcount_with['spam']:
                    wordcount_with['spam'][word] = 1
                else:
                    wordcount_with['spam'][word] += 1
    for doc in trained_data['notspam']:
        if trained_data['notspam'][doc].get(minpredword)<1:
            new_trained_data_without_word['notspam'][doc] = trained_data['notspam'][doc]
            for word in trained_data['notspam'][doc]:
                wordlist_without[word]=1
                if word not in wordcount_without['notspam']:
                    wordcount_without['notspam'][word] = 1
                else:
                    wordcount_without['notspam'][word] += 1
        else:
            new_trained_data_with_word['notspam'][doc] = trained_data['notspam'][doc]
            for word in trained_data['notspam'][doc]:
                wordlist_with[word]=1
                if word not in wordcount_with['notspam']:
                    wordcount_with['notspam'][word] = 1
                else:
                    wordcount_with['notspam'][word] += 1
    cantuselist.append(minpredword)
    for word in cantuselist:
        if word in wordlist_with:
            del wordlist_with[word]
        if word in wordlist_without:
            del wordlist_without[word]
    root = Tree()
    root.data = minpredword
    root.left = generateDecisionTree(new_trained_data_with_word,wordlist_with,wordcount_with)
    root.right = generateDecisionTree(new_trained_data_without_word,wordlist_without,wordcount_without)
    return root

# This method is used to walk on the decision tree and classify the documents based on the boolean value found on the leaf nodes.
def classifyDocs(trained_data, rootnode):
    rootofDT = rootnode
    spamcorrectcount = 0
    notspamcorrectcount = 0
    for topic in trained_data.keys():
        for doc in trained_data[topic].keys():
            while rootnode.isLeaf == False:
                if isPresent(trained_data[topic][doc],rootnode.data):
                    rootnode = rootnode.left
                else:
                    rootnode = rootnode.right
            if rootnode.data == True and topic == 'spam':
                spamcorrectcount += 1
            elif rootnode.data == False and topic == 'notspam':
                notspamcorrectcount += 1
            rootnode = rootofDT
    correctcount = spamcorrectcount + notspamcorrectcount
    total = len(trained_data['spam']) + len(trained_data['notspam'])
    print "Finished Classification\n"
    print "Percentage Accuracy: " + str(100.0*correctcount/total) + " %"
    print "Confusion Matrix:"
    print "_______________________________________________________"
    print "|                | Predicted Spam | Predicted Not-Spam|"
    print "|Actual Spam     | " + str(spamcorrectcount) + "           | " + \
    str(len(trained_data['spam'])-spamcorrectcount) + "                |"
    print "|Actual Not-Spam | " + str(len(trained_data['notspam'])-notspamcorrectcount) + \
    "             | " + str(notspamcorrectcount) + "              |"
