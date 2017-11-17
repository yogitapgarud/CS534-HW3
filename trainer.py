from __future__ import division
import numpy as np
from collections import defaultdict
import sys
from math import log
from tagger import decode
from tagger import mle
from tagger import readfile
from tagger import test
startsym, stopsym = "<s>", "</s>"

def devError(dictionary, model, filename):

    errorCount = 0
    total = 0

    for words, tags in readfile(filename):

        mytags = decode(words, dictionary, model)

        if tags != mytags:
            errorCount += 1
        total += 1

    return errorCount / total

def unavgPerceptron(dictionary, model, phixy, filename):

	currentEpoch = 1
	totalEpoch = 10  
	errors = 0
	#weightvector = np.zeros(len(model))
        best_dev_err = float("inf")

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0

    		for words, tags in readfile(filename):

                        count = 0
                        phixz = defaultdict(int)
		        mytags = decode(words, dictionary ,model)
                        #print mytags
                        last = 'DT'
                        phixz[startsym, last] = 1

                        for word, tag in zip(words, mytags):
                            phixz[word, tag] += 1
                            phixz[last, tag] += 1
                            last = tag

			if tags != mytags:

                                for k in phixy.keys():
                                        #print(k)
                                        c, w, t = k
                                        #print "c: ", c, "w: ", w, "t: ", t, "k: ", k
                                        model[w, t] += phixy[k]

                                for w, t in phixz:
                                        model[w, t] -= phixz[w, t]

				updates += 1
                                #print "updates: ", updates

        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
                        count += 1

                dev_err = devError(dictionary, model, devfile)
                train_err = updates / count

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
		features = sum(model.values())
                #print "features: ", features
                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))

def avgPerceptron(dictionary, model, filename):

	currentEpoch = 1
	totalEpoch = 5
	errors = 0
	#weightvector = np.zeros(len(model))
        best_dev_err = float("inf")

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0

    		for words, tags in readfile(filename):
		        mytags = decode(words, dictionary ,model)

			if tags != mytags:
				model = model + phixy[count] - phixz
				updates += 1

        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)

                dev_err = test(devfile, dictionary, model)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    print(weightvector)

                #print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
		features = np.count_nonzero(weightvector)
		print("epoch {%d}, updates {%d}, features {%d}, train_err {0:.2%}, dev_err {0:.2%}" %(currentepoch, updates, features, errors/tot * 100, dev_err))

if __name__ == "__main__":

	trainfile, devfile = sys.argv[1:3]
    
	dictionary, model, phixy = mle(trainfile)

	unavgPerceptron(dictionary, model, phixy, trainfile)

	print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	print "dev_err {0:.2%}".format(test(devfile, dictionary, model))


