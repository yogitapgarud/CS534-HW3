from collections import defaultdict

class myDefaultdict:
	def __init__(self, datatype):

		self.__mydict = defaultdict(datatype)

	def __add__(self, other):

		for w, v in other.iteritems():
			self.__mydict[w] = self.__mydict[w] + v

		return self.__mydict

	def addmult(self, other, multiplier):

		for w, v in other.iteritems():
			self.__mydict[w] = self.__mydict[w] + multiplier * v

		return self.__mydict


