import csv
import random
import math

class NaiveBayes(object):

	def __init__(self):
		# see https://en.wikipedia.org/wiki/Log_probability
		self.logEncode = False
		self.computeRatios = False
		pass

	#1. handle data

	def loadCsv(self, filename):
		lines = csv.reader(open(filename, "rb"))
		dataset = list(lines)
		for i in range(len(dataset)):
			dataset[i] = [float(x) for x in dataset[i]]
		return dataset

	#split samples into train and test sets (67% - 33%)

	def splitDataset(self, dataset, splitRatio):
		trainSize = int(len(dataset) * splitRatio)
		trainSet = []
		copy = list(dataset)
		while len(trainSet) < trainSize:
			index = random.randrange(len(copy))
			trainSet.append(copy.pop(index))
		return [trainSet, copy]
		pass
	
	# 2. Summarize data

	# 2.1 Separate Data by Class

	def separateByClass(self, dataset):
		separated = {}
		for i in range(len(dataset)):
			vector = dataset[i]
			#assume last attribute (-1) IS the CLASS VALUE
			if (vector[-1] not in separated):
				separated[vector[-1]] = []
			separated[vector[-1]].append(vector)
			pass
		return separated


	# 2.2 Calculate Mean

	def mean(self, numbers):
		return float(sum(numbers))/float(len(numbers))

	# 2.3 Calculate Standard Deviation
	def stdev(self, numbers):
		avg = self.mean(numbers)
		variance = sum ([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
		return math.sqrt(variance)

	# 2.4 Summarize dataset
	def summarize(self, dataset):
		# zip group values for each attribute into their own lists
		summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
		del summaries[-1] # last element represents the class
		return summaries
	# 2.5 Summarize Attributes By Class

	def summarizeByClass(self, dataset):
		separated = self.separateByClass(dataset)
		summaries = {}
		for classValue, instances in separated.iteritems():
			summaries[classValue] = self.summarize(instances)
		return summaries

	# 3. Make prediction
	# Plug known details into the Gaussian and check likelihood of attribute belonging to the class
	# 3.1 Calculate Gaussian Probability Density Function

	def calculateProbability(self, x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

	# 3.2 Calculate Class Probabilities

	# from https://gist.github.com/3991982.git
	def log_add(self, x, y):
		maximum = max(x,y)
		minimum = min(x,y)
		if(abs(maximum - minimum) > 30):
		# the difference is too small, return the just the maximum
			return maximum
		##return maximum + math.log(1 + math.pow(2, minimum - maximum), 2) 
		return maximum + math.log(1 + math.pow(2, minimum - maximum)) 


	def calculateClassProbabilities(self, summaries, inputVector):
		probabilities = {}
		for classValue, classSummaries in summaries.iteritems():
			probabilities[classValue] = 0 if self.logEncode else 1
			# print probabilities[classValue]
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				prob = self.calculateProbability(inputVector[i], mean, stdev) 
				if self.logEncode:
					probabilities[classValue] = self.log_add(probabilities[classValue], prob)
				else :
					probabilities[classValue] *= prob
				pass
			pass
		return probabilities

		# 3.3 Make prediction

	def predict(self, summaries, inputVector):
		probabilities = self.calculateClassProbabilities(summaries, inputVector)
		# print "class prob= %r"%probabilities
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.iteritems():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel
	
		# 3.4 Esttimate Accuracy

	# 4 Make Predictions

	def getPredictions(self, summaries, testSet):
		predictions = []
		for i in range(len(testSet)):
			result = self.predict(summaries, testSet[i])
			predictions.append(result)
			pass
		return predictions

		# 5 Get Accuracy

	def getAccuracy(self, testSet, predictions):
		correct = 0
		for x in range(len(testSet)):
			if testSet[x][-1] == predictions[x]:
				correct +=1
		return (correct/float(len(testSet))) * 100.0

	
	# the probability of a data instance belonging to one class, divided by the 
	# sum of the probabilities of the data instance belonging to each class. 
	# For example an instance had the probability of 0.02 for class A and 0.001 
	# for class B, the likelihood of the instance belonging 
	# to class A is (0.02/(0.02+0.001))*100 which is about 95.23%.

	def calculateClassInstanceProbabilities (self, summaries, testSet):
		probabilities = []
		# for each instance
		for t in range(len(testSet)):
			# calculate summation of probabilities each class i.e., calculateClassProbabilities
			cProbabilities = self.calculateClassProbabilities(summaries, testSet[t])
			sumProbabilities = 0

			if self.logEncode :
				for ix in range(len(cProbabilities)):
					sumProbabilities =  -math.log(math.exp(-sumProbabilities) + math.exp(-cProbabilities[ix]))
					pass
			else :
				sumProbabilities = sum (cProbabilities.values())

			# for each class in that result
			ratios = []
			for c in range(len(summaries)):
			# compute ratio of probabiliy in that class
				formatted = "{0:.2f}%".format((cProbabilities[c]/sumProbabilities)*100)
				ratios.append(formatted)
			probabilities.append(ratios);
		return probabilities


	def runExperiment(self, dataset, splitRatio):
		trainingSet, testSet = self.splitDataset(dataset, splitRatio)
		summaries = self.summarizeByClass(trainingSet)
		predictions =  self.getPredictions(summaries, testSet)

		return self.getAccuracy(testSet, predictions)

	def runMiniTests(self):

		#test load by checking row number
		filename = 'pima-indians-diabetes.csv'
		dataset = loadCsv(filename)
		print "Loaded data file %s with %d rows and %r features" % (filename, len(dataset), len(dataset[0]))
		
		dataset2 = [[1], [2], [3], [4], [5]]
		splitRatio = 0.67
		train, test = splitDataset(dataset2, splitRatio)
		print "\n-------- splitDataset mini-test (ratio = %r) ---------" % splitRatio
		print "Split %d rows into train %r and test with %r" % (len(dataset2), train, test)
		
		dataset3 = [[1,20,1], [2, 21, 0], [3, 22, 1]]
		separatd = separateByClass(dataset3);
		print "\n-------- separateByClass mini-test (set = %r) ---------" % dataset3
		print "Separared instances: %r" % separatd
		
		numbers = [1,2,3,4,5]
		print "\n-------- mean and stdev mini-test ---------"
		print "Summary of %r: mean=%r, stdev=%r" % (numbers, mean(numbers), stdev(numbers))
		
		
		dataset4 = [[1,20,0], [2, 21, 1], [3, 22, 0]]
		summary = summarize(dataset4)
		print "\n-------- Summarize mini-test %r ---------" % dataset4
		print "Attributes summaries: %r" % summary
		
		dataset5 = [[1,20,1], [2,21,0], [3, 22, 1], [4, 22, 0]]
		summary = summarizeByClass(dataset5)
		print "\n-------- Summarize by class mini-test %r ---------" % dataset5
		print "Summary by class value: %r" % summary

		x_value= 71.5
		mean_value = 73
		stdev_value = 6.2
		probability = calculateProbability(x_value, mean_value, stdev_value)
		print "\n-------- calculateProbabilitymini-test x=%r mean=%r stdev=%r ---------" % (x_value, mean_value, stdev_value)
		print "Probability of belonging to this class: %r" % probability


		summaries = {0: [(1, 0.5)], 1:[(20, 5.0)]}
		inputVector = [1.1, '?']
		probabilities = calculateClassProbabilities(summaries, inputVector)
		print "\n-------- calculateClassProbabilities mini-test ---------" 
		print "Probabilities for each class: %r" % probabilities
		
		summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
		inputVector =  [1.1, '?']
		result = predict(summaries, inputVector)
		print "\n-------- prediction mini-test summaries=%r---------" % summaries
		print "Prediction: %r" %result
		
		summaries = {'A': [(1, 0.5)], 'B': [(20, 5.0)]}
		testSet = [[1.1, '?'], [19.1, '?']]
		predictions = getPredictions(summaries, testSet)
		print "\n-------- getPredictions mini-test summaries=%r---------" % summaries
		print "Predictions: %r" % predictions
		
		testSet = [[1,1,1,'a'], [2,2,2, 'a'], [3,3,3, 'b']]
		predictions = ['a', 'a', 'a']
		accuracy = getAccuracy(testSet, predictions)
		print "\n-------- getAccuracy mini-test set=%r---------" % testSet
		print "Accuracy: %r" % accuracy


	def execute(self):
		filename = 'pima-indians-diabetes.csv'
		splitRatio = 0.67
		dataset = self.loadCsv(filename)
		trainingSet, testSet = self.splitDataset(dataset, splitRatio)
		print "Split %r rows in to train=%r and test=%r rows" % (len(dataset), len(trainingSet), len(testSet))
		print "Log Encode is", 'ON' if self.logEncode else 'OFF'

		summaries = self.summarizeByClass(trainingSet)
		# print "summaries=%r"%summaries
		predictions =  self.getPredictions(summaries, testSet)
		#print "predictions=%r"%predictions

		if not self.logEncode and self.computeRatios:
			pRatios = self.calculateClassInstanceProbabilities(summaries, testSet)
			print "> probability of data instances belonging to one class"
			print "No.  \t Class %r" % summaries.keys()
			for x in range(len(pRatios)):
				print "%r : \t %r"% (x, pRatios[x])
			
		accuracy = self.getAccuracy(testSet, predictions)
		print ('\n> Accuracy: {0}%').format(accuracy)

if __name__ == "__main__":
	print "\n", "*"*10,  "Main", "*"*10
	NaiveBayes().execute()
