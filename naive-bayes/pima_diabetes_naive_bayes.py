import csv
import random

#1. handle data

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#test load by checking row number
filename = 'pima-indians-diabetes.csv'
dataset = loadCsv(filename)
print "Loaded data file %s with %d rows and %r features" % (filename, len(dataset), len(dataset[0]))


#split samples into train and test sets (67% - 33%)

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
	pass

dataset2 = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset2, splitRatio)
print "-------- splitDataset mini-test -------"
print "Split %d rows into train %r and test with %r" % (len(dataset2), train, test)


#2. summarize data
