import numpy as np
from tagger import decode
from tagger import decodetrigram1
from trainer import avgPerceptron
from tagger import mle
from tagger import readfile
from trainer import avgPerceptronBivariant1
from trainer import avgPerceptronTrigramFeatures
import sys

def Predictor(model):
    with open("test.txt.lower.unk.unlabeled", "r") as f:
        output = open('test.txt.lower.unk.best.txt', 'w') 

        for line in f:
            
            words = line.split()

            #print(words)

            mytags = decodetrigram1(words, dictionary , model)

            words_tags = zip(words, mytags)

            for w, t in words_tags:

	            output.write(w)
                    output.write("/")
		    output.write(t)
                    output.write(" ")  

            output.write('\n')

        output.close()

def devPredictor(model, filename):
    with open("dev.txt.lower.unk", "r") as f:
        output = open('dev.txt.lower.unk.best', 'w') 

        for words, tags in readfile(filename):

            mytags = decodetrigram1(words, dictionary , model)

            words_tags = zip(words, mytags)

            for w, t in words_tags:

	            output.write(w)
                    output.write("/")
		    output.write(t)
                    output.write(" ")  

            output.write('\n')

        output.close()


if __name__ == "__main__":

	trainfile, devfile, featurefile = sys.argv[1:4]
    
	dictionary, model = mle(trainfile)
	#model = avgPerceptron(dictionary, model, trainfile, devfile, featurefile)
	#model = avgPerceptronBivariant1(dictionary, model, trainfile, devfile)
	model = avgPerceptronTrigramFeatures(dictionary, model, trainfile, devfile)
	print(type(model))
	Predictor(model)
	devPredictor(model, devfile)
