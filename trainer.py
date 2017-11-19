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
from mydict import myDefaultdict

startsym, stopsym = "<s>", "</s>"

def unavgPerceptron(dictionary, filename, devfile, totalEpoch = 10):

	currentEpoch = 1
        best_dev_err = float("inf")
	final_model = defaultdict(int)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))
	model = defaultdict(float)

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0
		count = 1
		c = 0

    		for words, tags in readfile(trainfile):

			c += 1 
                        mytags = decode(words, dictionary, final_model)

                        if tags != mytags:

				errorsentences += 1
				phidelta = defaultdict(float)
				wordseq = [startsym] + words + [stopsym]
				tagseq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

                                for i, (w, t1, t2) in enumerate(zip(wordseq, tagseq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						errors += 1

		                                

					if t1 != t2 or tagseq[i-1] != z[i-1]:
						phidelta[tagseq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1

                                for w, t in phidelta:
                                        model[w, t] += phidelta[w, t]

				
				updates += 1

        		#errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        		tot += len(words)
                        count += 1                

                dev_err = test(devfile, dictionary, model)
                train_err = errors / total
                #print(train_err)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                features = sum(v != 0 for _, v in model.iteritems())

                #for key, values in model.iteritems():
                #    if values != 0:
                #        features += 1

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100)
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))

def avgPerceptron(dictionary, trainfile, devfile, totalEpoch = 10 ):

	currentEpoch = 1
        best_dev_err = float("inf")
	modelAvg = defaultdict(float)
	countAvg = 1
	model = defaultdict(float)
	final_model = defaultdict(float)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0
		count = 1
		c = 0

    		for words, tags in readfile(trainfile):

			c += 1 
                        mytags = decode(words, dictionary, final_model)

                        if tags != mytags:

				errorsentences += 1
				phidelta = defaultdict(float)
				wordseq = [startsym] + words + [stopsym]
				tagseq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

                                for i, (w, t1, t2) in enumerate(zip(wordseq, tagseq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						errors += 1

		                                

					if t1 != t2 or tagseq[i-1] != z[i-1]:
						phidelta[tagseq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1

                                for w, t in phidelta:
                                        model[w, t] += phidelta[w, t]
					modelAvg[w, t] += countAvg * phidelta[w, t]
				 
				#model += phidelta 
				#modelAvg = modelAvg.addmult(phidelta, countAvg)
				
				updates += 1

			countAvg += 1
                        count += 1                

                for w, t in model:
                    final_model[w, t] = model[w, t] - modelAvg[w, t] / countAvg

                dev_err = test(devfile, dictionary, final_model)
                train_err = errors / total

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                features = sum(v != 0 for _, v in final_model.iteritems())

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100)

def avgPerceptronTrigramFeatures(dictionary, trainfile, devfile, totalEpoch = 10):

	currentEpoch = 1
	errors = 0
        best_dev_err = float("inf")
	model = defaultdict(float)
	modelAvg = defaultdict(int)
	countAvg = 1
	final_model = defaultdict(int)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x,y): len(x), trainset))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0
		count = 1

    		for words, tags in readfile(trainfile):

			mytags = decodetrigram(words, dictionary , final_model)

			if tags != mytags:

				errorsentences += 1
                        	phidelta = defaultdict(float)
				wordseq = [startsym] + words + [stopsym]
				tagseq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]
				count = 1

		                for i, (w, t1, t2) in enumerate(zip(words, tags, mytags)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						phidelta[tagseq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1
						errors += 1
						
					if i > 1 and (t1 != t2 or tagseq[i-1] != z[i-1] or tagseq[i-2] != z[i-2]):
						
						phidelta[tagseq[i-2], tagseq[i-1], t1] += 1
						phidelta[z[i-2], z[i-1], t2] -= 1
						phidelta[tagseq[i-1], t1, w] += 1
						phidelta[tagseq[i-1], t2, w] -= 1
					count += 1

                                for w, v in phidelta.iteritems():
                                        model[w] += v			#phixy[count][w]
					modelAvg[w] += countAvg * v	#phixy[count][w]

				updates += 1
                                #print "updates: ", updates

			countAvg += 1
                        count += 1

                for w in model:
                    final_model[w] = model[w] - modelAvg[w] / countAvg        

                #dev_err = test(devfile, dictionary, final_model)

		

                train_err = errors / total
                #print(train_err)

		errors = tot = 0
		for words, tags in readfile(devfile):
			mytags = decode(words, dictionary , final_model)
			errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
			tot += len(words) 
        
		dev_err = errors / tot

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                features = sum(v != 0 for _, v in final_model.iteritems())

                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))
                #print("train_err {0:.2%} dev_err {0:.2%}".format(errors / tot * 100, dev_err))

def avgPerceptronBivariant1(dictionary, trainfile, devfile, totalEpoch = 10 ):

	currentEpoch = 1
        best_dev_err = float("inf")
	modelAvg = defaultdict(float)
	countAvg = 1
	model = defaultdict(float)
	final_model = defaultdict(float)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0
		count = 1
		c = 0

    		for words, tags in readfile(trainfile):

			c += 1 
                        mytags = decode(words, dictionary, final_model)

                        if tags != mytags:

				errorsentences += 1
				phidelta = defaultdict(float)
				wordseq = [startsym] + words + [stopsym]
				tagseq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

                                for i, (w, t1, t2) in enumerate(zip(wordseq, tagseq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						errors += 1

					if t1 != t2 or tagseq[i-1] != z[i-1]:
						phidelta[tagseq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1
						phidelta[tagseq[i-1], w] += 1
						phidelta[z[i-1], w] -= 1

                                for w, t in phidelta:
                                        model[w, t] += phidelta[w, t]
					modelAvg[w, t] += countAvg * phidelta[w, t]
				
				updates += 1

			countAvg += 1
                        count += 1                

                for w, t in model:
                    final_model[w, t] = model[w, t] - modelAvg[w, t] / countAvg

                dev_err = test(devfile, dictionary, final_model)
                train_err = errors / total

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                features = sum(v != 0 for _, v in final_model.iteritems())

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100)

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

	return phixy

if __name__ == "__main__":

	trainfile, devfile = sys.argv[1:3]
    
	dictionary, model, phi = mle(trainfile)

	print "Unaveraged structured Perceptron: "
	unavgPerceptron(dictionary, trainfile, devfile)

	print "Averaged structured Perceptron:"
	avgPerceptron(dictionary, trainfile, devfile)

	print "Averaged structured Perceptron with Trigram t-2 t-1 t0:"
	avgPerceptronTrigramFeatures(dictionary, trainfile, devfile)

	print "Averaged structured Perceptron with Bigram variatn with t-1 w:"
	avgPerceptronBivariant1(dictionary, trainfile, devfile)
	#print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	#print "dev_err {0:.2%}".format(test(devfile, dictionary, model))


