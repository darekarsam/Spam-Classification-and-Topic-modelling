'''
---------------------------------------------------------------------------------------------------
PART 1 - NAIVE BAYES
---------------------------------------------------------------------------------------------------
(Apart form the below description, the files bayesmodel.py and filereader.py in modules folder contain additional comments which can help in your understanding)
1. Problem Formulation
    The formulation of the Naive Bayes Model is pretty simple. It assumes that every word is independent of the other given the topic. We proceed with this assumption.
    A topic-wise dictionary is kept which contains all the word frequencies for words occurring in documents of that topic. This two level dictionary structure helps to 
    quickly do the lookups when calculating the posterior probabilities.
2. How the program works?
    The program builds this dictionary structure along with other counts which help to speed up the calculation of posterior probabilities for a test document.
    Topic with this highest probability is the label assigned to the test document.
3. Design decision
    As mentioned before we use a dictionary of dictionaries to speed up our lookups.
4. Performance

    Here are the confusion matrices
    Binary Features (Accuracy = 98.2%)
             notspam  spam
    notspam     1360     9
    spam          37  1148

    Continuous Features (Accuracy = 98.79%)
             notspam  spam
    notspam     1359    10
    spam          21  1164

    Accuracy is not significantly different when dealing with binary or continuous features. However we did observe that the top 10 word printed were very good
    for the continuous features case. Here they are for your reference:

    --------------------------------------------------
    Top 10 words associated with spam
    --------------------------------------------------
    content-type - 1076
    0100 - 1005
    1.0 - 1001
    mime-version - 998
    8.11.6/8.11.6 - 914
    smtp - 855
    127.0.0.1 - 832
    content-transfer-encoding - 827
    localhost - 825
    postfix - 807

    --------------------------------------------------
    Top 10 words associated with notspam
    --------------------------------------------------
    8.11.6/8.11.6 - 1353
    0100 - 1347
    dogma.slashnull.org - 1342
    127.0.0.1 - 1292
    localhost - 1290
    postfix - 1289
    delivered-to - 1283
    imap - 1278
    ist - 1275
    fetchmail-5.9.0 - 1275


---------------------------------------------------------------------------------------------------
PART 2 - DECISION TREE
---------------------------------------------------------------------------------------------------
(1) a description of how you formulated the problem, including precisely defining the abstractions.
Here we have the problem of learning a decision tree.
We have the words found in the document as the features for the decision tree.
For each word we then calculate the "Avg. Disorder." or the "Entropy"
This can be done using two ways either by checking the presence of the word or the count of the word.
Once we find the feature or the word which splits the dataset giving the least entropy.
We then make a recursive call on the split up datasets to further split them based on the word with the least entropy.
This uses the ID3 algorithm for the binary feature type.
We use binning for the continuous type where we calculate the frequency/total ratio of the counts of the words in a document.

(2) a brief description of how your program works.

In the training mode we read the all the documents in the given training dataset.
It is stored in a dictionary which has two keys 'spam' and 'notspam'.
These keys have another dictionary as its values having the file name in the dataset as the keys.
This dictionary will have a defaultdict of type float as its value.
This defaultdict contains "word":count as its value.
We assume that words which are not there in the defaultdict have value 0.

The class 'Tree' is used to store the nodes of the Decision Tree.
This is the data structure for the decision tree nodes.
data contains the word or will contain boolean True/False for leaf nodes.
True indicates a spam Document and False indicates a non-spam document.

The generateDecisionTree method is called recursively to split the dataset based on the word with the least frequency.
The words found here are then stored in the objects of the Tree type.
The tree structure generated is then stored to the model file using pickle.

In the testing mode we then read the tree structure using pickle.
This rootnode is used to classify the documents from the test set by doing a walk/traversal of the tree.

(3) a discussion of any problems, assumptions, simplications, and/or design decisions you made.
We have kept a word count which stores the count of the number of spam/notspam documents the word is found in.
So we don't have to calculate the no of documents the word is present in for each feature/word.
We have kept an additional parameter at the end 'b' or 'c' which will choose the binary feature or continuous feature.

(4) answers to any questions asked below in the assignment.
The binary feature gave better result as compared to the continuous feature for the decision tree as the model tries to overfit.
Thus the binary feature was better for the decision tree.
Here we found that based purely on the accuracy the bayes model with continuous feature gives us the best result.

---------------------------------------------------------------------------------------------------
Results:
---------------------------------------------------------------------------------------------------
1)Decision Tree with binary feature
The top 4 levels of Decision Tree are as follows:
                                x-spam-status:

                        False                           in-reply-to:

                                        False                   zzzz@localhost.netnoteinc.com

                                                        False                           reserved.
Percentage Accuracy: 98.6296006265 %
Confusion Matrix:
_______________________________________________________
|                | Predicted Spam | Predicted Not-Spam|
|Actual Spam     | 1173           | 12                |
|Actual Not-Spam | 23             | 1346              |

**********
In this representation the root node is "x-spam-status:"
"x-spam-status:" has left node as "False" and right node as "in-reply-to:"
"in-reply-to:" has left node as "False" and right node as "zzzz@localhost.netnoteinc.com"
"zzzz@localhost.netnoteinc.com" has left node as "False" and right node as "reserved."
More nodes further down the tree are not shown here
Similar representation is used below.
**********

2)Decision Tree with continuous feature
The top 4 levels of Decision Tree are as follows:
                                x-spam-status:(0.01)

                        False(None)                             zzzz@localhost.netnoteinc.com(0.01)

                                        False(None)                     in-reply-to:(0.01)

                                                        False(None)                             reserved.(0.01)
Percentage Accuracy: 97.9639780736 %
Confusion Matrix:
_______________________________________________________
|                | Predicted Spam | Predicted Not-Spam|
|Actual Spam     | 1142           | 43                |
|Actual Not-Spam | 9             | 1360               |
'''
import sys

if len(sys.argv) < 5:
    print "Too few arguments"
    print "The program expects arguments in this fashion"
    print "python spam.py [mode] [technique] [dataset-directory] [model-file]"
    print "[mode] - ""train"" or ""test"""
    print "[model-file] - filename to save model to, or load model from"
    print "[technique] - can be ""dt"" decision tree or ""bayes"" naive bayes model"
else:
    # mode can be 'train' or 'test'
    mode = str(sys.argv[1])
    # technique can be 'bayes' or 'dt'
    technique = str(sys.argv[2])
    # dataset_directory is of the form 'test' or 'train' or 'part1/train' or 'part1/test'
    dataset_directory = str(sys.argv[3])
    # model_file can be named anything as long as it is same for train and test and the feature used
    model_file =  str(sys.argv[4])
    # feature can be 'b' for binary or 'c' for continuous feature
    try:
        feature =  str(sys.argv[5])
    except IndexError:
        feature = None

    if technique == "dt":
        if feature == 'b' or feature == None:
            from modules.spam_dt_binary import *
        elif feature == 'c':
            from modules.spam_dt_cont import *
        if mode == "train":
            print "Started reading data"
            (trained_data,wordslist,wordscount) = readData(dataset_directory)
            print "Finished reading data"
            print "\nStarted learning decision tree"
            rootnode = generateDecisionTree(trained_data,wordslist,wordscount)
            print "Finished learning decision tree"
            print_tree3(rootnode)
            with open(model_file, 'wb') as output:
                pickle.dump(rootnode, output, pickle.HIGHEST_PROTOCOL)
            print "Finished writing learned decision tree to model file"
        if mode == "test":
            print "Started reading data"
            (trained_data,wordslist,wordscount) = readData(dataset_directory)
            print "Finished reading data"
            print "Started reading decision tree from model file"
            with open(model_file, 'rb') as input:
                rootnode = pickle.load(input)
            print_tree3(rootnode)
            print "Finished reading decision tree from model file"
            print "Started classification"
            classifyDocs(trained_data, rootnode)

    elif technique == "bayes":
        import modules.filereader as fr
        import modules.bayesmodel as bm
        import os
        import pickle

        stopwords_dict = fr.stopwords()
        if mode == "train":
            if os.path.exists(model_file):
                os.remove(model_file)

            data = fr.readdata(dataset_directory, stopwords_dict)

            # change the featuretype variable for changing the representation ["binary" or "frequency"]
            featuretype = "binary" if feature == 'b' else "frequency"
            trained_model = bm.train(data, featuretype=featuretype, part="part1")
            pickle.dump(trained_model, open(model_file, "wb"))
            print "Model written to", model_file
            print "Use the same filename for running on test set"

        elif mode == "test":
            if os.path.exists(model_file):
                trained_model = pickle.load(open(model_file, "rb"))
                bm.test(trained_model, dataset_directory, stopwords_dict)
            else:
                print "Model file", model_file, "does not exist"
