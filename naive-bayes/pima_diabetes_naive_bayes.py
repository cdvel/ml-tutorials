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
print "\n-------- splitDataset mini-test (ratio = %r) ---------" % splitRatio
print "Split %d rows into train %r and test with %r" % (len(dataset2), train, test)


# 2. Summarize data

# 2.1 Separate Data by Class

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		#assume last attribute (-1) IS the CLASS VALUE
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
		pass
	return separated

dataset3 = [[1,20,1], [2, 21, 0], [3, 22, 1]]
separatd = separateByClass(dataset3);
print "\n-------- separateByClass mini-test (set = %r) ---------" % dataset3
print "Separared instances: %r" % separatd


# 2.2 Calculate Mean

import math 

def mean(numbers):
	return sum(numbers)/float(len(numbers))

# 2.3 Calculate Standard Deviation
def stdev(numbers):
	avg = mean(numbers)
	variance = sum ([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

numbers = [1,2,3,4,5]
print "\n-------- mean and stdev mini-test ---------"
print "Summary of %r: mean=%r, stdev=%r" % (numbers, mean(numbers), stdev(numbers))

# 2.4 Summarize dataset

def summarize(dataset):
	# zip group values for each attribute into their own lists
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1] # last element represents the class
	return summaries

dataset4 = [[1,20,0], [2, 21, 1], [3, 22, 0]]
summary = summarize(dataset4)
print "\n-------- Summarize mini-test %r ---------" % dataset4
print "Attributes summaries: %r" % summary

# 2.5 Summarize Attributes By Class

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

dataset5 = [[1,20,1], [2,21,0], [3, 22, 1], [4, 22, 0]]
summary = summarizeByClass(dataset5)
print "\n-------- Summarize by class mini-test %r ---------" % dataset5
print "Summary by class value: %r" % summary

# 3. Make prediction
# 
# Plug known details into the Gaussian and check likelihood of attribute belonging to the class

# 3.1 Calculate Gaussian Probability Density Function

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

x_value= 71.5
mean_value = 73
stdev_value = 6.2
probability = calculateProbability(x_value, mean_value, stdev_value)
print "\n-------- calculateProbabilitymini-test x=%r mean=%r stdev=%r ---------" % (x_value, mean_value, stdev_value)
print "Probability of belonging to this class: %r" % probability

# 3.2 Calculate Class Probabilities

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev) 
			pass
		pass
	return probabilities

summaries = {0: [(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print "\n-------- calculateClassProbabilities mini-test ---------" 
print "Probabilities for each class: %r" % probabilities

# 3.3 Make prediction

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 

summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
inputVector =  [1.1, '?']
result = predict(summaries, inputVector)
print "\n-------- prediction mini-test summaries=%r---------" % summaries
print "Prediction: %r" %result

# 3.4 Esttimate Accuracy

# 4 Make Predictions

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
		pass
	return predictions

summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
testSet = [[1.1, '?'], [19.1, '?']]
predictions = getPredictions(summaries, testSet)
print "\n-------- getPredictions mini-test summaries=%r---------" % summaries
print "Predictions: %r" % predictions

# 5 Get Accuracy

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct +=1
	return (correct/float(len(testSet))) * 100.0

testSet = [[1,1,1,'a'], [2,2,2, 'a'], [3,3,3, 'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print "\n-------- getAccuracy mini-test set=%r---------" % testSet
print "Accuracy: %r" % accuracy


def main():
	filename = 'pima-indians-diabetes.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print "Split %r rows in to train=%r and test=%r rows" % (len(dataset), len(trainingSet), len(testSet))
	print trainingSet[0]
	summaries = summarizeByClass(trainingSet)
	predictions =  getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print ('Accuracy: {0}%').format(accuracy)


print "\n*********** Main ***********"
main()