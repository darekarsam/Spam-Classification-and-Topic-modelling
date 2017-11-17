"""
(Apart form the below description, the files bayesmodel.py and filereader.py in modules folder contain additional comments which can help in your understanding)

1. Problem Formulation
	The formulation of the Naive Bayes Model is pretty simple. It assumes that every word is independent of the other given the topic. We proceed with this assumption.
	A topic-wise dictionary is kept which contains all the word frequencies for words occurring in documents of that topic. This two level dictionary structure helps to 
	quickly do the lookups when calculating the posterior probabilities.
2. How the program works?
	The program builds this dictionary structure along with other counts which help to speed up the calculation of posterior probabilities for a test document.
	Topic with this highest probability is the label assigned to the test document. Now when the fraction given the program via command line argument is < 1, EM comes
	into the picture. Here is how EM works briefly in our case

	E step - assumes the topic assignment for each document is fixed and then calculates the word likelihood ie. P(W|T) over all topics
			 When assigning the labels in the first iteration (only), we use random.choice() and if this value is less than fraction, we look at the actual label,
			 else we randomly assign one of the twenty topics.
	M step - assumes the word likelihoods are fixed and then calculates the posterior i.e P(T|w1 w2 .. wn) where w1 .. wn belong to a document D.
			 This is done for all documents D in the train dataset.

	We repeat these EM step 3 times, since we found this is enough according to our experiments.
3. Design decision
	As mentioned before we use a dictionary of dictionaries to speed up our lookups. We also keep a document to topic assignment dictionary for the iterative
	steps in the EM algorithm
4. Accuracies for different values of the fraction
	Fraction					Average Accuracy (%)
	----------------------------------------------
	0.0								5.97
	0.01							15.63
	0.1								66.85
	0.2								75.72
	0.3								78.41
	0.4								79.26
	0.5								79.7
	0.8								79.9
	1.0								80.03

	As we can see when the fraction is 0.0 the accuracy is almost as good as random assignment. We have not peeked at the actual labels here. However, when we have some
	some labelled training data, we can see the accuracy rising and then plateauing after 0.3. Here are some sample confusion matrices:

	Confusion matrix (Fraction = 0.2)
	--------------------------------------------------------------------------
	             atheism  autos  baseball  christian  crypto  electronics
	atheism          232      0         0         27       3            0   
	autos              0    329         0          1       2            5   
	baseball           1      2       343          2       4            1   
	christian          4      0         1        361       0            0   
	crypto             1      0         0          0     350            2   
	electronics        2      6         1          0      26          246   
	forsale            0     15         4          2       6           13   
	graphics           1      3         3          2      16           14   
	guns               0      0         0          2       5            1   
	hockey             0      0         8          0       2            0   
	mac                0      6         0          0       5           12   
	medical            0      8         1          8       5            6   
	mideast            0      0         0          7       0            0   
	motorcycles        1     13         0          1       0            5   
	pc                 0      1         0          0       9           32   
	politics           8      0         0          6       4            0   
	religion          41      0         0         34       2            1   
	space              0      3         0          1       7            5   
	windows            1      2         0          9      23            8   
	xwindows           0      2         0          0      11            3   

	             forsale  graphics  guns  hockey  mac  medical  mideast  \
	atheism            0         1     4       1    0        2        7   
	autos             10         1     2       1    1        2        2   
	baseball           1         0     1      23    1        4        2   
	christian          0         1     3       1    0        3        1   
	crypto             0         1    13       2    1        8        2   
	electronics        9        10     1       1    6       16        1   
	forsale          239         3     0       5   23        6        3   
	graphics           4       261     0       0    9        5        1   
	guns               0         1   314       0    0        4        5   
	hockey             1         0     0     377    0        1        1   
	mac               18        15     2       2  255       11        1   
	medical            3         5     2       1    0      316        5   
	mideast            0         1     0       0    0        1      359   
	motorcycles        1         0     3       0    0        0        3   
	pc                20        12     0       2   24        0        0   
	politics           0         0    68       0    0        5        7   
	religion           0         0    19       0    2        2       10   
	space              2         2     1       0    1        5        3   
	windows            8        50     2       1   16        5        2   
	xwindows           7        45     3       0    4        1        0   

	             motorcycles   pc  politics  religion  space  windows  xwindows  
	atheism                0    2        13        23      3        0         1  
	autos                 23    0         8         0      8        1         0  
	baseball               2    1         6         0      3        0         0  
	christian              1    1         1        19      1        0         0  
	crypto                 1    4         5         2      2        0         2  
	electronics           16   21         2         1     20        7         1  
	forsale                7   35         5         3     16        3         2  
	graphics               3   12         4         2     11        6        32  
	guns                   1    0        19         9      3        0         0  
	hockey                 2    0         6         1      0        0         0  
	mac                    6   26         2         0     10        7         7  
	medical                2    3        19         2      8        1         1  
	mideast                0    0         8         0      0        0         0  
	motorcycles          370    1         0         0      0        0         0  
	pc                     0  259         1         0      2       23         7  
	politics               1    0       193         9      9        0         0  
	religion               0    0        10       122      8        0         0  
	space                  0    2         7         1    350        0         4  
	windows                1   62        18         2      9      138        37  
	xwindows               3   12         0         0     12        3       289  


	Confusion matrix (Fraction = 1.0)
	--------------------------------------------------------------------------
	             atheism  autos  baseball  christian  crypto  electronics  \
	atheism          246      0         0         24       2            0   
	autos              0    348         0          1       2            4   
	baseball           1      2       356          2       1            0   
	christian          3      0         1        367       0            0   
	crypto             0      1         2          0     362            1   
	electronics        1      6         0          0      22          269   
	forsale            0     15         2          0       3            9   
	graphics           1      0         3          2      13           10   
	guns               0      0         0          1       5            1   
	hockey             0      0         2          0       3            0   
	mac                0      1         0          0       6           12   
	medical            0      7         0          6       5            6   
	mideast            6      0         0          3       0            0   
	motorcycles        0     14         0          0       0            0   
	pc                 0      0         0          0       3           33   
	politics           8      1         1          1       4            0   
	religion          40      1         0         22       2            0   
	space              4      1         0          2       7            5   
	windows            2      0         0          7      15            6   
	xwindows           0      1         1          0       9            4   

	             forsale  graphics  guns  hockey  mac  medical  mideast  \
	atheism            0         0     2       1    0        1        8   
	autos             10         0     0       1    2        2        0   
	baseball           2         0     1      19    0        3        1   
	christian          0         1     1       1    0        3        1   
	crypto             2         3     8       1    3        4        1   
	electronics        6        10     0       0    6       15        1   
	forsale          278         4     0       5   14        4        2   
	graphics           7       276     0       0   10        4        1   
	guns               0         0   312       0    0        3        1   
	hockey             2         0     0     387    0        0        0   
	mac               10        15     0       1  287        2        0   
	medical            5         5     1       1    0      331        4   
	mideast            0         0     1       1    0        0      355   
	motorcycles        2         0     2       0    0        1        1   
	pc                15         8     0       0   17        0        0   
	politics           0         0    52       0    0        7       10   
	religion           0         1     9       0    0        3        4   
	space              1         2     2       1    0        7        0   
	windows           12        51     1       0   15        5        0   
	xwindows           4        42     0       1    2        1        0   

	             motorcycles   pc  politics  religion  space  windows  xwindows  
	atheism                0    2         3        26      3        0         1  
	autos                 21    0         4         0      0        1         0  
	baseball               2    0         4         0      3        0         0  
	christian              1    0         4        14      0        1         0  
	crypto                 0    1         2         1      2        1         1  
	electronics            6   22         2         1     16        9         1  
	forsale                5   29         1         1     12        3         3  
	graphics               0   10         1         0     12        6        33  
	guns                   1    0        19        18      3        0         0  
	hockey                 1    0         3         1      0        0         0  
	mac                    2   25         1         0     11        7         5  
	medical                1    3        11         2      6        1         1  
	mideast                0    0        10         0      0        0         0  
	motorcycles          376    1         0         0      1        0         0  
	pc                     0  285         0         0      1       25         5  
	politics               0    0       204        14      8        0         0  
	religion               0    0         8       156      5        0         0  
	space                  0    1         5         0    352        0         4  
	windows                2   55        13         2      7      171        30  
	xwindows               2    8         0         0      4        6       310  

"""

import modules.filereader as fr
import modules.bayesmodel as bm
import sys
import pickle
import os

if len(sys.argv) < 4:
	print "Too few arguments"
	print "The program expects arguments in this fashion"
	print "python topics.py [mode] [dataset-directory] [model-file] [fraction]"
	print "[mode] - ""train"" or ""test"""
	print "[model-file] - filename to save model to, or load model from"
	print "[fraction] - required with train, values between 0.0 - 1.0. Default is 1.0"
else:
	mode = str(sys.argv[1])	
	directory = str(sys.argv[2])
	model_file = str(sys.argv[3])	
	fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
	
	stopwords_dict = fr.stopwords()
	if mode == "train":
		if os.path.exists(model_file):
			os.remove(model_file)
		
		data = fr.readdata(directory, stopwords_dict)

		# change the featuretype variable for changing the representation ["binary" or "frequency"]
		trained_model = bm.train(data, featuretype="frequency" ,fraction=fraction, part="part2")
		pickle.dump(trained_model, open(model_file, "wb"))
		print "Model written to", model_file
		print "Use the same filename for running on test set"

	elif mode == "test":
		if os.path.exists(model_file):
			trained_model = pickle.load(open(model_file, "rb"))
			bm.test(trained_model, directory, stopwords_dict)
		else:
			print "Model file", model_file, "does not exist"