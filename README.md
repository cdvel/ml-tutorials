# ml-tutorials

1. [k-nearest neighbors](http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)

  - Finds the N most similar elements based on euclidean, hammington  etc distance
  * instance-based: 
    - model the problem using data instances(rows) in order to make predictive decisions. 
    - In kNN all observations are retained as part of our model = extreme instance-based

  * competitive learning: 
    - internally model elements(instance) compete in order to make a predictive decision.
    - objective similarity measure between instances causes each instance to compete to be most similar to a given unseen data instance and contribute

  * lazy learning
    - The algorithm doesn't build a model until the time of prediction is required
    - Only relevant data to the unseen data (localized model)
    - Can be computationally expensive to repeat over larger training sets

  - Makes no assumption over the data, only that a distance measure can be calculated. Non-parametric or non-linear, doesn't assume functional form.

2. [naive bayes] (http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)  
