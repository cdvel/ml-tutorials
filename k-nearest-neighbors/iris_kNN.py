# import csv
# with open('iris.data', 'rb') as csvfile:
# 	lines = csv.reader(csvfile)
# 	for row in lines:
# 		print ', '.join(row)

#1. handle data

import csv
import random 

# this function loads a file and splits randomly 
# into train and test given a ratio
def loadDataset(filename, split, trainingSet=[], testSet=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)	#convert lines to 2dim array
		for x in range(len(dataset)):
			for y in range(4):
				#print "%d, %d, %s"%(x,y, dataset[x][y])
				dataset[x][y] = float(dataset[x][y]) #cast strings to floats
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
				pass
			pass
	pass

trainingSet = []
testSet = []
loadDataset('iris.data', 0.66, trainingSet, testSet)
print 'Train: ' + repr(len(trainingSet))
print 'Test:  ' + repr(len(testSet))
print 'Total: ' + repr(len(testSet)+len(trainingSet))


#2. similarity

import math

#used to calculate similarity given numeric values
def euclideanDistance(instance1, instance2, length):
	distance = 0;
	for x in range(length):
		distance+= pow((instance1[x] - instance2[x]), 2)
		pass
	return math.sqrt(distance)
	pass

#do test this function
# data1 = [2, 2, 2, 'a']
# data2 = [4, 4, 4, 'b']
# distance = euclideanDistance(data1, data2, 3)
# print 'Distance: ' + repr(distance)

#3. Neighbours

import operator

def getNeighbours(trainingSet, testInstance, k):
	distances = []
	#compute length
	length = len(testInstance) - 1
	#compute distances and store for all training set instances
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
		pass
	#sort them	
	distances.sort(key = operator.itemgetter(1))
	#select k-first from ordered set
	neighbours = []
	for x in range(k):
		neighbours.append(distances[x][0])
		pass
	return neighbours
	pass

# uncomment below to test code to this point

# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b'], [4.5, 4.5, 4.5, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbours = getNeighbours(trainSet, testInstance, 1)
# print neighbours

#4. response

def getResponse(neighbours):
	classVotes = {}
	for x in range(len(neighbours)):
		response = neighbours[x][-1]
		if response in classVotes:
			classVotes[response] +=1
		else:
			classVotes[response] = 1
		pass
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]	
	pass

# uncomment below to test code to this point

# neighbours = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
# response = getResponse(neighbours)
# print response

# #returns 1 response, if draw.

#5. accuracy

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	pass

# uncomment below to test code to this point

# testSet = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)

# ************ tying all together

def main():
	#prepare data
	trainingSet = []
	testSet = []
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	#generate predictions
	predictions = []
	k = 3
	for x in range(len(testSet)):
		neighbours = getNeighbours(trainingSet, testSet[x], k)
		result = getResponse(neighbours)
		predictions.append(result)
		print "> predicted = "+repr(result)+"\t, actual = "+repr(testSet[x][-1])
	
	accuracy = getAccuracy(testSet, predictions)
	print "Accuracy: %r" %accuracy,
	print "%"
		
	pass

main()