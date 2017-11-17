from __future__ import division
import numpy as np
from collections import defaultdict
import sys
from math import log
from tagger import decode
from tagger import mle
from tagger import readfile
from tagger import test

def perceptron(dictionary, model, filename):

	currentEpoch = 1
	totalEpoch = 5
	errors = 0
	weightvector = np.zeros(len(model))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0

    		for words, tags in readfile(filename):
		        mytags = decode(words, dictionary ,model)

			if tags != mytags:
				weightvector = weightvector + tags - mytags
				updates += 1

        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
		
		features = np.count_nonzero(weightvector)
		print("epoch {%d}, updates {%d}, features {%d}, train_err {0:.2%}, dev_err {0:.2%}" %(currentepoch, updates, features, errors/tot * 100, tot))

	
	

if __name__ == "__main__":
	trainfile, devfile = sys.argv[1:3]
    
	dictionary, model = mle(trainfile)

	perceptron(dictionary, model, trainfile)

	print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	print "dev_err {0:.2%}".format(test(devfile, dictionary, model))


