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

  * The Model 
    - A summary of data in the training set
    1. mean
    2. standard deviation for each (no. attributes * class values) 
    3. calculates probability of a specific attribute belonging to each class value      

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
    - Encoding using [Log Probabilities](https://en.wikipedia.org/wiki/Log_probability):
      - Reduces risk of floating point underflow (values too small to be represented)
      - More efficient by using summation of log probabilities instead of product of probabilities
      - Observations with the Iris dataset: 
        - Less accurate 
        - Faster
        - Log = (0.5s - 0.8s, accuracy=66%) vs Prob = (0.6 - 0.8, accuracy=74%)
        - Numerical stability (accuracy is more consistent)
    - using categorical data (ratios)
    - numeric attributes (with normal distribution)

  - [How to get best from Naive bayes] (http://machinelearningmastery.com/better-naive-bayes/) 