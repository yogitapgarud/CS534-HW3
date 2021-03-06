#git #!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import sys
from math import log
from mydict import myDefaultdict
startsym, stopsym = "<s>", "</s>"

def readfile(filename):
    for line in open(filename):
        wordtags = map(lambda x: x.rsplit("/", 1), line.split())
        yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair
    
def mle(filename): # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda : defaultdict(int))
    ttfreq = defaultdict(lambda : defaultdict(int)) 
    tagfreq = defaultdict(int)    
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):
        
        last = startsym
        tagfreq[last] += 1

        for word, tag in zip(words, tags) + [(stopsym, stopsym)]:
            #if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1            
            ttfreq[last][tag] += 1
            dictionary[word].add(tag)
            tagfreq[tag] += 1
            last = tag

    model = defaultdict(float)
    num_tags = len(tagfreq)

    for tag, freq in tagfreq.iteritems(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].iteritems():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2 # +1 smoothing
    
    #print "model len : ", len(model), "len dictionary : ", len(dictionary)
    #print(dictionary)     
    return dictionary, model

def mleTrigram(filename):

    tttfreq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    ttwfreq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    twfreq = defaultdict(lambda: defaultdict(int))
    dictionary = defaultdict(set)
    tagtagfreq = defaultdict(lambda: defaultdict(int))
    tagfreq = defaultdict(int)

    for words, tags in readfile(filename):

        lasttolast = startsym
        last = startsym
        tagtagfreq[startsym][startsym] += 1

        for word, tag in zip(words, tags) + [(stopsym, stopsym)]:

            twfreq[tag][word] += 1
            #ttfreq[last][tag] += 1
            ttwfreq[lasttolast][last][word] += 1
            tttfreq[lasttolast][last][tag] += 1
            dictionary[word].add(tag)
            tagtagfreq[last][tag] += 1
            tagfreq[tag] += 1
            lasttolast = last
            last = tag

    model = defaultdict(float)
    num_tags = len(tagtagfreq)

    for tag1 in tagtagfreq:
	for tag2 in tagtagfreq[tag1]:
	        logfreq = log(tagtagfreq[tag1][tag2] + num_tags)
		
        
		for t in tttfreq[tag1][tag2]:
			logf = log(tttfreq[tag1][tag2][t] + 1)
			model[tag1, tag2, t] = logf - logfreq

			for word in ttwfreq[tag1][tag2]:
			    #print("ttw: ", ttwfreq[tag1, tag2, word])
			    logw = log(ttwfreq[tag1][tag2][word])
			    model[tag1, tag2, word] = logw - logfreq

    num_tags = len(tagfreq)

    for tag, freq in tagfreq.iteritems(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].iteritems():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(tagtagfreq[tag][t] + 1) - logfreq2

    return dictionary, model

def decode(words, dictionary, model):

    def backtrack(i, tag):
        if i == 0:
            return []
        return backtrack(i-1, back[i][tag]) + [tag]

    words = [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    best[0][startsym] = 1
    back = defaultdict(dict)

    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])

    for i, word in enumerate(words[1:], 1):
        for tag in dictionary[word]:
            #print "tag ", tag, "for", word
            for prev in best[i-1]:
		#print "prev : ", prev, "i - 1 : ", i - 1
                score = best[i-1][prev] + model[prev, tag] + model[tag, word] + model['\p', prev, word] + model['\w', tag, words[i-1]]
                if score > best[i][tag]:
                    best[i][tag] = score
                    back[i][tag] = prev

    #   print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, stopsym)[:-1]
    #print "mytags : ", mytags
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags

def decodetrigram1(words, dictionary, model):

    def backtrack(i, tag):
        if i == 1:
            return []
        return backtrack(i-1, back[i][tag]) + [tag]

    words = [startsym] + [startsym] + words + [stopsym]

    best = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    best[0][startsym] = 1
    best[1][startsym] = 1

    back = defaultdict(dict)

    #for tag in dictionary[words[1]]:
	#best[1][tag] = 1
	#back[1][tag] = startsym

    #print " ".join("%s/%s" % wordtag for wordtag in zip(words,tags)[1:-1])

    #print "sentence : ", words
    for i, word in enumerate(words[2:], 2):
        for tag in dictionary[word]:
            #print "tag ", tag, "for", word
            for prev in best[i-1]:
		#print "prev : ", prev, "i - 1 : ", i - 1
		for lasttolast in best[i-2]:
			
			#print(model['\p', prev, word])
		        score = best[i-1][prev] + best[i-2][lasttolast]  + model[prev, tag] + model[tag, word]  + model['\p', prev, word] + model[lasttolast, prev, tag] #+ model[lasttolast, prev, word]  + model['\pt', prev, tag, word]  + model[lasttolast, prev, word] # #+ model['\w-1', tag, words[i-1]]  # #   # + model['\w', tag, words[i-1]] # #  # + model['\p', prev, word]    # #  #+ model[prev, tag, word] #   #	 	 model['\ww', words[i-1], word]+ model['\pp', lasttolast, word]	+ model[tag, words[i-1], word]

		        if score > best[i][tag]:
		            best[i][tag] = score
		            back[i][tag] = prev

    #   print i, word, dictionary[word], best[i]
    #print best[len(words)-1][stopsym]
    mytags = backtrack(len(words)-1, stopsym)[:-1]
    #print "mytags : ", mytags
    #print " ".join("%s/%s" % wordtag for wordtag in mywordtags)
    return mytags


def test(filename, dictionary, model):    
    
    errors = tot = 0
    for words, tags in readfile(filename):
        mytags = decode(words, dictionary ,model)
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot
        
if __name__ == "__main__":

    trainfile, devfile = sys.argv[1:3]
    
    dictionary, model, phixy = mle(trainfile)
                                                                             
    print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
    print "dev_err {0:.2%}".format(test(devfile, dictionary, model))
