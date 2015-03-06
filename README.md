# ml-tutorials

1. [k-Nearest neighbors](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

  - Finds the N most similar elements based on euclidean, hammington  etc distance

  * Instance-based
    - model the problem using data instances(rows) in order to make predictive decisions. 
    - In kNN all observations are retained as part of our model = extreme instance-based

  * Competitive learning 
    - internally model elements(instance) compete in order to make a predictive decision.
    - objective similarity measure between instances causes each instance to compete to be most similar to a given unseen data instance and contribute

  * Lazy learning
    - The algorithm doesn't build a model until the time of prediction is required
    - Only relevant data to the unseen data (localized model)
    - Can be computationally expensive to repeat over larger training sets

  - Makes no assumption over the data, only that a distance measure can be calculated. Non-parametric or non-linear, doesn't assume functional form.

2. [Naive bayes] (http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)  

  - Suits classification problems. Uses probabilities of each attribute belonging to each class to make a prediction

  - Fast and effective supervised learning for probabilistic prediction

  * Assumptions
    - Independent probabilities between attributes of a given classs
    - Numerical attributes are normally distributed 

  * Conditional probability
    - Probability of a class given a value of an attribute
    - The product of _conditional probabilities_ for each attribute = probability of an instance belonging to a class

  * Prediction
    - Calculate probabilities of instance belonging to class
    - Pick class with highest probability

  * Implementations
    - using categorical data (ratios)
    - numeric attributes (with normal distribution)

  - [How to get best from Naive bayes] (http://machinelearningmastery.com/better-naive-bayes/)
 