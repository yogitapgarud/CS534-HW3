from __future__ import division
import numpy as np
from collections import defaultdict
import sys
from math import log
from tagger import decode
from tagger import mle
from tagger import readfile
from tagger import test
from tagger import decodetrigram

startsym, stopsym = "<s>", "</s>"

def devError(dictionary, model, filename):

    errorCount = 0
    total = 0

    errors = tot = 0

    for words, tags in readfile(filename):

        mytags = decode(words, dictionary, model)

        #if tags != mytags:
        #    errorCount += 1

	errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words)

        total += 1

    return errors / tot

def unavgPerceptron(dictionary, model, phi, filename):

	currentEpoch = 1
	totalEpoch = 10  
	errors = 0
	#weightvector = np.zeros(len(model))
        best_dev_err = float("inf")

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0
		count = 1

    		for words, tags in readfile(filename):

                        phixz = defaultdict(int)
                        phixy = defaultdict(int)
		        mytags = decode(words, dictionary ,model)
                        #print mytags
                        last = 'DT'
                        phixz[startsym, last] = 1

                        for word, tag in zip(words, mytags):
                            phixz[word, tag] += 1
                            phixz[last, tag] += 1
                            last = tag

                        last = 'DT'
                        phixy[startsym, last] += 1

                        for word, tag in zip(words, tags):
                            phixy[word, tag] += 1
                            phixy[last, tag] += 1
                            last = tag

			if tags != mytags:

                                for w, t in phixy:
                                        #print(k)
                                        #w, t = k
                                        #print "c: ", c, "w: ", w, "t: ", t, "k: ", k
                                        #print "phixy [ ", w, t, " ] : ", phixy[w, t]
                                        model[w, t] += phixy[w, t]

                                for w, t in phixz:
                                        #print "phixz", w, t, ": ", phixz[w, t]
                                        model[w, t] -= phixz[w, t]

				updates += 1
                                #print "updates: ", updates

        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
                        count += 1

                dev_err = devError(dictionary, model, devfile)
                train_err = errors / tot
                #print(train_err)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
		#features = sum(1 if value != 0 for key, value in model.iteritems())
                features= 0

                for key, values in model.iteritems():
                    if values != 0:
                        features += 1

                #print "features: ", features
                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))

def avgPerceptron(dictionary, model, filename):
	currentEpoch = 1
	totalEpoch = 10  
	errors = 0
	#weightvector = np.zeros(len(model))
        best_dev_err = float("inf")
	modelAvg = defaultdict(int)
	countAvg = 1

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0
		count = 1

    		for words, tags in readfile(filename):

                        phixz = defaultdict(int)
                        phixy = defaultdict(int)
		        mytags = decode(words, dictionary ,model)
                        #print mytags
                        last = 'DT'
                        phixz[startsym, last] = 1

                        for word, tag in zip(words, mytags):
                            phixz[word, tag] += 1
                            phixz[last, tag] += 1
                            last = tag

                        last = 'DT'
                        phixy[startsym, last] += 1

                        for word, tag in zip(words, tags):
                            phixy[word, tag] += 1
                            phixy[last, tag] += 1
                            last = tag

			if tags != mytags:

                                for w, t in phixy:
                                        #print(k)
                                        #w, t = k
                                        #print "c: ", c, "w: ", w, "t: ", t, "k: ", k
                                        #print "phixy [ ", w, t, " ] : ", phixy[w, t]
                                        model[w, t] += phixy[w, t]
					modelAvg[w, t] += countAvg * phixy[w, t]

                                for w, t in phixz:
                                        	#print "phixz", w, t, ": ", phixz[w, t]
                                        model[w, t] -= phixz[w, t]
					modelAvg[w, t] -= countAvg * phixz[w, t]

				updates += 1
                                #print "updates: ", updates

			countAvg += 1
        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
                        count += 1

		#final_model = model - modelAvg / count
                #final_model = for w, t in model return model[w, t] - modelAvg[w, t] / countAvg
                final_model = defaultdict(int)

                for w, t in model:
                    final_model[w, t] = model[w, t] - modelAvg[w, t] / countAvg
                    

                dev_err = devError(dictionary, final_model, devfile)
                train_err = errors / tot
                #print(train_err)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
		#features = sum(1 if value != 0 for key, value in model.iteritems())
                features= 0

                for key, values in model.iteritems():
                    if values != 0:
                        features += 1

                #print "features: ", features
                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))

def avgPerceptronFeatures(dictionary, model, filename):
	currentEpoch = 1
	totalEpoch = 10  
	errors = 0
	#weightvector = np.zeros(len(model))
        best_dev_err = float("inf")
	modelAvg = defaultdict(int)
	countAvg = 1

	phixy, phixz = genfeatures(trainfile)

	for currentEpoch in range(1, totalEpoch + 1):
		
		errors = tot = 0
		updates = 0
		count = 1

    		for words, tags in readfile(filename):

                        mytags = decodetrigram(words, dictionary ,model)

			if tags != mytags:

                                for w, v in phixy[count].iteritems():
                                        model[w] += v			#phixy[count][w]
					modelAvg[w] += countAvg * v	#phixy[count][w]

                                for w, v in phixz[count].iteritems():
                                        	#print "phixz", w, t, ": ", phixz[w, t]
                                        model[w] -= v			#phixz[count][w]
					modelAvg[w] -= countAvg * v	#phixz[count][w]

				updates += 1
                                #print "updates: ", updates

			countAvg += 1
        		errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
                        count += 1

		#final_model = model - modelAvg / count
                #final_model = for w, t in model return model[w, t] - modelAvg[w, t] / countAvg
                final_model = defaultdict(int)

                for w in model:
                    final_model[w] = model[w] - modelAvg[w] / countAvg        

                dev_err = devError(dictionary, final_model, devfile)
                train_err = errors / tot
                #print(train_err)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
		#features = sum(1 if value != 0 for key, value in model.iteritems())
                features= 0

                for key, values in model.iteritems():
                    if values != 0:
                        features += 1

                #print "features: ", features
                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))


def genfeatures(filename):
	#f = open('myfile', 'w+')
	phixy = defaultdict(lambda : defaultdict(int))
	phixz = defaultdict(lambda : defaultdict(int))

	for words, tags in readfile(filename):

		        mytags = decode(words, dictionary ,model)
                        #print mytags
                        last = 'DT'
                        #phixz[startsym, last] = 1

			count = 1

                        for word, tag in zip(words, mytags):

				if count == 1:
					tagprev2 = last
					last = tag

				else:	
					#f.write
					phixz[count][tagprev2, last, tag] += 1
					phixz[count][last, word, tag] += 1
					tagprev2 = last
					last = tag
				count += 1

                        last = 'DT'
                        #phixy[startsym, last] += 1

			count = 1
                        for word, tag in zip(words, tags):

                           	if count == 1:
					tagprev2 = last
					last = tag

				else:	
					#f.write
					phixy[count][tagprev2, last, tag] += 1
					phixy[count][last, word, tag] += 1
					tagprev2 = last
					last = tag
				count += 1

	return phixy, phixz

if __name__ == "__main__":

	trainfile, devfile = sys.argv[1:3]
    
	dictionary, model, phi = mle(trainfile)

	print "Unaveraged structured Perceptron: "
	unavgPerceptron(dictionary, model, phi, trainfile)

	print "Averaged structured Perceptron:"
	avgPerceptron(dictionary, model, trainfile)

	print "Averaged structured Perceptron:"
	avgPerceptronFeatures(dictionary, model, trainfile)

	#print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	#print "dev_err {0:.2%}".format(test(devfile, dictionary, model))


