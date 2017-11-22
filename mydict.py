from collections import defaultdict

class myDefaultdict(defaultdict):

	def __init__(self, datatype):

		super(myDefaultdict).__init__(datatype)

		#type(self)
		#self.__mydict = defaultdict(datatype)

	def __add__(self, other):

		for w in other:
			self[w] = self[w] + other[w]

	def addmult(self, other, multiplier):

		#print("other: ", other)
		#print("multiplier : ", multiplier)

		for w, v in other:
			self[w] = self[w] + multiplier * other[w]


