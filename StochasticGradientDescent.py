# -*- coding: utf-8 -*-
"""
This code is mostly forom:
    https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
"""

import numpy as np


class StochasticGradientDescent:
    
    def __init__(self):
        self.lossHistory = []
        
    def fit(self, X, y, epochs=100, alpha=0.001, batch_size=128):
        self.W = np.random.uniform(size=(X.shape[1],))/100
        self.params = {'epochs':epochs, 'alpha':alpha, 'batch_size':batch_size}
        
        # loop over the desired number of epochs
        for each in np.arange(0, epochs):
            # initialize the total loss for the epoch
            epochLoss = []
            
            # loop over our data in batches
            for (batchX, batchY) in self._next_batch(X, y, batch_size):
                # take the dot product between our current batch of 
                # features and weight matrix 'W', then pass this value
                # through the sigmoid activation function
                preds = self.predict(batchX)
                
                # now that we have our predictions, we need to determine
                # our 'error', which is the difference between our predictons
                # and the true values
                error = preds - batchY
                
                # given our 'error', we can compute the total loss value on
                # the batch as the sum of squared loss
                loss = np.sum(error ** 2)
                epochLoss.append(loss)
                
                # the gradient update is therefore the dot product between
                # the transpose of our current batch and the error on the # batch
                gradient = 2 * batchX.T.dot(error) / batchX.shape[0]
                
                # use the gradient computed on the current batch to take
                # a "step" in the corrent direction
                self.W += - alpha * gradient
            
            # update our loss history list by taking the average loss
            # across all batches
            self.lossHistory.append(np.average(epochLoss))
            
    def predict(self, X):
        return self._linear_activation(X.dot(self.W))
        
    def _linear_activation(self, x):
        """
        compute and return the simple linear sum value for a given input x
        """
        return x
    
    def _next_batch(self, X, y, batchSize):
        # loop over our dataset 'X' in mini-batches of size 'batchSize'
        for i in np.arange(0, X.shape[0], batchSize):
            # yield a tuple of the current batched data and labels
            yield (X[i:i + batchSize], y[i:i + batchSize])
    

if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs

    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)

    sgd = StochasticGradientDescent()    
    sgd.fit(X, y)    
    sgd.predict(X)
    
    for i in range(X.shape[0]):
        print(y[i], sgd.predict(X[i]))
