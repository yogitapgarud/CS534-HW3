from __future__ import division
import numpy as np
from collections import defaultdict
import sys
import time
import matplotlib.pyplot as plt
from math import log
from tagger import decode
from tagger import mle
from tagger import readfile
from tagger import test
from tagger import decodetrigram1
from mydict import myDefaultdict

startsym, stopsym = "<s>", "</s>"

def unavgPerceptron(dictionary, filename, devfile, totalEpoch = 10):

	currentEpoch = 1
        best_dev_err = float("inf")
	countAvg = 1
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))
	totalTime = 0
	#startTime = time.time()
        epochs = []
        trainArr = []
        devArr = []
	model = defaultdict(float)

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		count = 1
		c = 0
		updates = 0
		startTime = time.time()

    		for words, tags in readfile(trainfile):

			c += 1 
                        mytags = decode(words, dictionary, model)

                        if tags != mytags:

				errorsentences += 1
				phidelta = defaultdict(int)
				w_seq = [startsym] + words + [stopsym]
				t_seq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

                                for i, (w, t1, t2) in enumerate(zip(w_seq, t_seq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						errors += 1

		                                

					if t1 != t2 or t_seq[i-1] != z[i-1]:
						phidelta[t_seq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1

                                for w, v in phidelta.iteritems():
                                        model[w] += phidelta[w]
				 
				updates += 1
				#model += phidelta 
				#modelAvg = modelAvg.addmult(phidelta, countAvg)

			countAvg += 1
                        count += 1                

		endTime = time.time()
		totalTime += endTime - startTime

                dev_err = test(devfile, dictionary, model)
                train_err = errors / total

                epochs.append(currentEpoch)
                trainArr.append(train_err * 100)
                devArr.append(dev_err * 100)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #features = sum(v != 0 for _, v in model.iteritems())
		features = 0
		for i, v in model.iteritems():
			if v != 0:
				features += 1

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100)

	#endTime = time.time()
	print("UnAverage Perceptron time : ", totalTime)
	plt.plot(epochs, trainArr, label='Train Error')
        plt.plot(epochs, devArr, label='Dev Error')
        plt.title('Error Rates for Unaveraged and Averaged Perceptron')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
	plt.show()

def avgPerceptron(dictionary, trainfile, devfile, featurefile, totalEpoch = 10):

	currentEpoch = 1
        best_dev_err = float("inf")
	modelAvg = defaultdict(float)
	countAvg = 1
	model = defaultdict(float)
	final_model = defaultdict(float)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))
	totalTime = 0
	epochs = []
        trainArr = []
	avgtrainArr = []
        devArr = []
	avgdevArr = []

	for currentEpoch in range(1, totalEpoch + 1):
		
		startTime = time.time()
		errorsentences = errors = tot = 0
		updates = 0
		count = 1
		c = 0

    		for words, tags in readfile(trainfile):

			c += 1 
                        mytags = decode(words, dictionary, model)
			#print(type(model))

                        if tags != mytags:

				#print(type(model))
				errorsentences += 1
				phidelta = defaultdict(float)
				w_seq = [startsym] + words + [stopsym]
				t_seq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

				features = []

				#for line in open(featurefile):
				#	line = line.strip()
				#        templates = line.split("_")	#map(lambda x: x.split("_"), line.split())
				#	print("templates: ", templates)
				        #features.add(yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair)

                                for i, (w, t1, t2) in enumerate(zip(w_seq, t_seq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1
						errors += 1

		                                

					if t1 != t2 or t_seq[i-1] != z[i-1]:
						phidelta[t_seq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1

                                for w, v in phidelta.iteritems():
                                        model[w] += phidelta[w]
					modelAvg[w] += countAvg * phidelta[w]
			
				#model += phidelta 
				#modelAvg = model.addmult(phidelta, countAvg)
				
				updates += 1

			countAvg += 1
                        count += 1                

                for w, v in model.iteritems():
                    final_model[w] = model[w] - modelAvg[w] / countAvg

		endTime = time.time()
		totalTime += endTime - startTime

                dev_err = test(devfile, dictionary, model)
                avg_dev_err = test(devfile, dictionary, final_model)
                train_err = errors / total
		avg_train_err = test(trainfile, dictionary, final_model)

		epochs.append(currentEpoch)
                trainArr.append(train_err * 100)
		avgtrainArr.append(avg_train_err * 100)
                devArr.append(dev_err * 100)
                avgdevArr.append(avg_dev_err * 100)

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #features = sum(v != 0 for _, v in final_model.iteritems())

		features = 0

		for i, v in final_model.iteritems():
			if v != 0:
				features += 1

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%, avg_dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100, avg_dev_err * 100)

	#endTime = time.time()
	print("Average Perceptron time : ", totalTime)

	plt.plot(epochs, trainArr, label='Train Error')
	plt.plot(epochs, avgtrainArr, label='Average Train Error')
        plt.plot(epochs, devArr, label='Dev Error')
        plt.plot(epochs, avgdevArr, label='Avg Dev Error')
        plt.title('Error Rates for Unaveraged and Averaged Perceptron')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
	plt.show()
	return final_model

def avgPerceptronTrigramFeatures(dictionary, model, trainfile, devfile, totalEpoch = 8):

	currentEpoch = 1
	errors = 0
        best_dev_err = float("inf")
	#model = defaultdict(float)
	modelAvg = defaultdict(int)
	countAvg = 1
	final_model = defaultdict(int)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x,y): len(x), trainset))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0

    		for words, tags in readfile(trainfile):

			mytags = decodetrigram1(words, dictionary , final_model)

			if tags != mytags:

				errorsentences += 1
                        	phidelta = defaultdict(float)
				w_seq = [startsym] + [startsym] + words + [stopsym]
				t_seq = [startsym] + [startsym] + tags + [stopsym]
				z = [startsym] + [startsym] + mytags + [stopsym]

		                for i, (w, t1, t2) in enumerate(zip(w_seq, t_seq, z)[2:], 2):

					#phidelta['\ww', w_seq[i-1], w] += 1

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1

						phidelta[t_seq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1

						#phidelta['\w-1', t1, w_seq[i-1]] += 1
						#phidelta['\w-1', t2, w_seq[i-1]] -= 1
						errors += 1
						
					if t1 != t2 or t_seq[i-1] != z[i-1] or t_seq[i-2] != z[i-2]:

						phidelta[t_seq[i-2], t_seq[i-1], t1] += 1
						phidelta[z[i-2], z[i-1], t2] -= 1

						#phidelta[t_seq[i-1], t1, w_seq[i-1], w] += 1
						#phidelta[z[i-1], t2, w_seq[i-1], w] -= 1

						#phidelta[t_seq[i-2], t_seq[i-1], w] += 1
						#phidelta[z[i-2], z[i-1], w] -= 1

						#phidelta['\pt', t_seq[i-1], t1, w] += 1
						#phidelta['\pt', z[i-1], t2, w] -= 1

						phidelta['\p', t_seq[i-1], w] += 1
						phidelta['\p', z[i-1], w] -= 1

						#phidelta['\pp', t_seq[i-2], w] += 1
						#phidelta['\pp', z[i-2], w] -= 1
						
						#phidelta['\w', t1, w_seq[i-1]] += 1
						#phidelta['\w', t2, w_seq[i-1]] -= 1

						#phidelta[t1, w_seq[i-1], w] += 1
						#phidelta[t2, w_seq[i-1], w] -= 1


                                for w, v in phidelta.iteritems():
                                        model[w] += v			
					modelAvg[w] += countAvg * v	

				updates += 1

			countAvg += 1

                for w, v in model.iteritems():
                    final_model[w] = model[w] - modelAvg[w] / countAvg        

                dev_err = test(devfile, dictionary, final_model)

                train_err = errors / total
                #print(train_err)
     
                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #features = sum(v != 0 for _, v in final_model.iteritems())
		features = 0
		for i, v in final_model.iteritems():
			if v != 0:
				features += 1

                print("epoch {0}, updates {1}, features {2}, train_err {3:.2%}, dev_err {4:.2%}".format(currentEpoch, updates, features, train_err, dev_err))

	return final_model

def avgPerceptronBivariant1(dictionary, model, trainfile, devfile, totalEpoch = 10):

	currentEpoch = 1
        best_dev_err = float("inf")
	modelAvg = defaultdict(float)
	countAvg = 1
	#model = defaultdict(float)
	final_model = defaultdict(float)
	trainset = list(readfile(trainfile))
	total = sum(map(lambda (x, y): len(x), trainset))

	for currentEpoch in range(1, totalEpoch + 1):
		
		errorsentences = errors = tot = 0
		updates = 0
		count = 1

    		for words, tags in readfile(trainfile):

                        mytags = decode(words, dictionary, final_model)

                        if tags != mytags:

				errorsentences += 1
				phidelta = defaultdict(float)
				w_seq = [startsym] + words + [stopsym]
				t_seq = [startsym] + tags + [stopsym]
				z = [startsym] + mytags + [stopsym]

                                for i, (w, t1, t2) in enumerate(zip(w_seq, t_seq, z)[1:], 1):

					if t1 != t2:
						phidelta[t1, w] += 1
						phidelta[t2, w] -= 1						
						phidelta[t_seq[i-1], t1] += 1
						phidelta[z[i-1], t2] -= 1
						phidelta['\w', t1, w_seq[i-1]] += 1
						phidelta['\w', t2, w_seq[i-1]] -= 1
						phidelta['\p', t_seq[i-1], w] += 1
						phidelta['\p', z[i-1], w] -= 1
						errors += 1

                                for w, v in phidelta.iteritems():
                                        model[w] += phidelta[w]
					modelAvg[w] += countAvg * phidelta[w]
				
				updates += 1

			countAvg += 1
                        count += 1                

                for w, v in model.iteritems():
                    final_model[w] = model[w] - modelAvg[w] / countAvg

                dev_err = test(devfile, dictionary, final_model)
                train_err = errors / total

                if dev_err < best_dev_err:
                    best_dev_err = dev_err
                    #print "model : ", model

                #features = sum(v != 0 for _, v in final_model.iteritems())
		features = 0
		for i, v in final_model.iteritems():
			if v != 0:
				features += 1

                print "epoch %d, updates %d, |W| = %d, train_err %.2f%%, dev_err %.2f%%" % (currentEpoch, updates, features, train_err * 100, dev_err * 100)

	return final_model

if __name__ == "__main__":

	trainfile, devfile, featurefile = sys.argv[1:4]
    
	dictionary, model = mle(trainfile)

	#print "Unaveraged structured Perceptron: "
	#unavgPerceptron(dictionary, trainfile, devfile)

	#print "Averaged structured Perceptron:"
        #avgPerceptron(dictionary, trainfile, devfile, featurefile)
	#print(len(model))

	#dictionary, model = mleTrigram(trainfile)
	print "Averaged structured Perceptron with Trigram best t-2 t-1 t0:"
	avgPerceptronTrigramFeatures(dictionary, model, trainfile, devfile)

	#print "Averaged structured Perceptron with Trigram t-2 t-1 t0:"
	#avgTrigram(dictionary, model, trainfile, devfile)

	print "Averaged structured Perceptron with Bigram variant with t-1 w:"
	avgPerceptronBivariant1(dictionary, model, trainfile, devfile)
	#print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
	#print "dev_err {0:.2%}".format(test(devfile, dictionary, model))


