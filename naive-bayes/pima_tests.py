import time
from pima_diabetes_naive_bayes import NaiveBayes

verbose = False
nbayes = NaiveBayes()
dataset = nbayes.loadCsv('pima-indians-diabetes.csv');

print "\n", "*"*10,  "Tests", "*"*10

nruns = 100
accuracies ={}
avg = 0;

start_time = time.time()
print "\n", "Log Encode OFF"
nbayes.logEncode = False

if verbose :
	print " %r  \t %r "% ("run", "accuracy")

for x in range(nruns):
	accuracies[x] = nbayes.runExperiment(dataset, splitRatio = 0.67)
	if verbose:
		print " %r : \t %r "% (x, accuracies[x])
	avg += accuracies[x]

avg /= nruns
print "%r runs; average accuracy = %r [%r secs]"% (nruns, avg, time.time()-start_time)

start_time = time.time()
print "\n", "Log Encode ON"
nbayes.logEncode = True

if verbose :
	print " %r  \t %r "% ("run", "accuracy")

for x in range(nruns):
	accuracies[x] = nbayes.runExperiment(dataset, splitRatio = 0.67)
	if verbose:
		print " %r : \t %r "% (x, accuracies[x])
	avg += accuracies[x]

avg /= nruns
print "%r runs; average accuracy = %r [%r secs]"% (nruns, avg, time.time()-start_time)
